import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.npu_attention import NPUMultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

# 时间步嵌入模块
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)  
        self.attention = NPUMultiheadAttention(channels, num_heads=8, batch_first=True)
    
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        batch, C, H, W = x_norm.shape
        x_flat = x_norm.view(batch, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(batch, C, H, W)
        
        return residual + attn_out


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        # 修正：GroupNorm需要两个参数
        # self.bnorm1 = nn.GroupNorm(32, out_channels) 
        # self.bnorm2 = nn.GroupNorm(32, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t):
        # 添加时间步嵌入
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]  # 扩展维度以匹配特征图
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = x + time_emb
        x = self.relu(self.bnorm2(self.conv2(x)))
        return x


# 下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super(DownBlock, self).__init__()
        self.use_attention = use_attention
        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_attention:
            self.attention = AttentionBlock(out_channels)

    def forward(self, x, t):
        x = self.conv_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        p = self.pool(x)
        return x, p


# 上采样块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super(UpBlock, self).__init__()
        self.use_attention = use_attention
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim)  # 注意：这里是in_channels，不是out_channels
        if use_attention:
            self.attention = AttentionBlock(out_channels)

    def forward(self, x, skip, t):
        x = self.upconv(x)
        # 确保上采样后的特征图与跳跃连接的特征图尺寸匹配
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        return x


# U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256, cond_channels=2, num_fourier_freq=10):
        super(UNet, self).__init__()
        
        fourier_channels = 2 * num_fourier_freq
        self.in_channels = in_channels
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.GELU(),
            nn.Linear(time_emb_dim*4, time_emb_dim),
        )
        
        # 傅里叶编码参数
        self.num_fourier_freq = num_fourier_freq
        self.fourier_channels = fourier_channels
        
        # 下采样路径 - 在更深层使用注意力
        self.down1 = DownBlock(self.in_channels, 64, time_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim)  # 中等分辨率加注意力
        self.down4 = DownBlock(256, 512, time_emb_dim)  # 较高分辨率加注意力

        self.bottleneck = ConvBlock(512, 1024, time_emb_dim)
        self.atte_bottleneck = AttentionBlock(1024)  # 瓶颈层注意力
        
        # 上采样路径 - 在浅层也使用注意力
        self.up1 = UpBlock(1024, 512, time_emb_dim)
        self.up2 = UpBlock(512, 256, time_emb_dim)
        self.up3 = UpBlock(256, 128, time_emb_dim)
        self.up4 = UpBlock(128, 64, time_emb_dim)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, timestep):
        x_cond = x
        # 生成时间步嵌入
        t = self.time_mlp(timestep)
        
        # 下采样路径
        d1, p1 = self.down1(x_cond, t)
        d2, p2 = self.down2(p1, t)
        d3, p3 = self.down3(p2, t)
        d4, p4 = self.down4(p3, t)

        # 瓶颈层
        b = self.bottleneck(p4, t)
        b = self.atte_bottleneck(b)
        
        # 上采样路径
        u1 = self.up1(b, d4, t)
        u2 = self.up2(u1, d3, t)
        u3 = self.up3(u2, d2, t)
        u4 = self.up4(u3, d1, t)

        output = self.final_conv(u4)
        return output