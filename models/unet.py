import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np  # 添加numpy导入
from models.npu_attention import NPUMultiheadAttention


# ==================== 时间嵌入 ====================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置嵌入"""
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


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=60):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div_term)
#         pe[:, 1::2] = torch.cos(pos * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)

# ==================== 基础卷积块 ====================
class ConvBlock(nn.Module):
    """基础卷积块（用于下采样模块）"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 时间嵌入
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, t):
        # 第一层卷积
        x = self.relu(self.bnorm1(self.conv1(x)))
        
        # 时间嵌入
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]  # 扩展维度以匹配图像特征
        x = x + time_emb
        
        # 第二层卷积
        x = self.relu(self.bnorm2(self.conv2(x)))
        
        return x


# ==================== 自注意力模块 ====================
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


# ==================== 跨模态交叉注意力 ====================
class CrossModalCrossAttention(nn.Module):
    """
    跨模态交叉注意力：太阳风向量 ↔ 极光图像特征
    
    关键思想：
    1. 太阳风向量作为"全局上下文"
    2. 图像特征作为"局部查询"
    3. 让图像特征"关注"相关的太阳风信息
    """
    
    def __init__(self, 
                 image_channels,      # 图像特征通道数
                 wind_channels,       # 太阳风特征维度（编码后）
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.image_channels = image_channels
        self.wind_channels = wind_channels
        self.num_heads = num_heads
        
        # ==================== 图像特征处理（查询） ====================
        # 将图像特征投影为查询
        self.image_to_query = nn.Conv2d(image_channels, image_channels, 1)
        
        # ==================== 太阳风向量处理（键/值） ====================
        # 将太阳风向量投影为键和值
        self.wind_to_key = nn.Linear(wind_channels, image_channels)
        self.wind_to_value = nn.Linear(wind_channels, image_channels)
        
        # ==================== NPU兼容的多头注意力机制 ====================
        self.multihead_attn = NPUMultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== 输出投影 ====================
        self.output_proj = nn.Conv2d(image_channels, image_channels, 1)
        
        # ==================== 极光物理偏置 ====================
        # 根据太阳风参数调整注意力偏置（可选）
        self.physics_bias_proj = nn.Sequential(
            nn.Linear(wind_channels, image_channels),
            nn.ReLU(),
            nn.Linear(image_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image_features, wind_features):
        """
        image_features: [B, C_img, H, W] - 图像特征
        wind_features: [B, C_wind] - 太阳风特征向量
        """
        B, C_img, H, W = image_features.shape
        
        # ==================== 1. 准备查询 ====================
        # 图像特征 -> 查询
        query = self.image_to_query(image_features)  # [B, C_img, H, W]
        query_flat = query.view(B, C_img, -1).transpose(1, 2)  # [B, H*W, C_img]
        
        # ==================== 2. 准备键和值 ====================
        # 太阳风向量 -> 键/值
        # 太阳风是全局条件，我们为每个空间位置创建相同的键/值
        key = self.wind_to_key(wind_features)  # [B, C_img]
        value = self.wind_to_value(wind_features)  # [B, C_img]
        
        # 扩展为序列长度1
        key = key.unsqueeze(1)  # [B, 1, C_img]
        value = value.unsqueeze(1)  # [B, 1, C_img]
        
        # ==================== 3. 注意力计算 ====================
        # 查询: [B, H*W, C_img]
        # 键: [B, 1, C_img]
        # 值: [B, 1, C_img]
        attn_output, _ = self.multihead_attn(
            query=query_flat, 
            key=key, 
            value=value,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False
        )  # [B, H*W, C_img]
        
        # ==================== 4. 恢复形状 ====================
        attn_output = attn_output.transpose(1, 2).view(B, C_img, H, W)
        
        # ==================== 5. 输出投影 + 残差 ====================
        output = self.output_proj(attn_output)
        output = image_features + output
        
        return output


class SolarWindEncoder(nn.Module):
    """太阳风参数编码器：7维向量 -> 高层次特征"""
    
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=256):
        super().__init__()
        
        # 物理分组编码：磁场、速度、压力
        self.magnetic_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # Bx, By, Bz
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # V
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.pressure_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # P
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # 融合编码器
        self.fusion_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU()
        )
        
    def forward(self, solar_wind):
        """
        solar_wind: [B, 7] - [Bx, By, Bz, V, P]
        """
        # 分离不同物理量
        magnetic = solar_wind[:, 0:3]  # Bx, By, Bz
        velocity = solar_wind[:, 3:4]  # V
        pressure = solar_wind[:, 4:5]  # P
        
        # 分别编码
        magnetic_feat = self.magnetic_encoder(magnetic)  # [B, hidden_dim//2]
        velocity_feat = self.velocity_encoder(velocity)  # [B, hidden_dim//2]
        pressure_feat = self.pressure_encoder(pressure)  # [B, hidden_dim//4]
        
        # 拼接所有特征
        combined = torch.cat([magnetic_feat, velocity_feat, pressure_feat], dim=1)
        
        # 融合编码
        output = self.fusion_encoder(combined)  # [B, output_dim]
        
        return output


class CrossModalConvBlock(nn.Module):
    """带有跨模态交叉注意力的卷积块"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, wind_dim,
                 use_cross_attention=True):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # 基础卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 时间嵌入
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 跨模态交叉注意力（可选，根据需求设置为False）
        if use_cross_attention:
            self.cross_attention = CrossModalCrossAttention(
                image_channels=out_channels,
                wind_channels=wind_dim,
                num_heads=8
            )
    
    def forward(self, x, t, wind_features=None):
        # 第一层卷积
        x = self.relu(self.bnorm1(self.conv1(x)))
        
        # 时间嵌入
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        
        # 第二层卷积
        x = self.relu(self.bnorm2(self.conv2(x)))
        
        # 跨模态注意力（如果启用且有风特征）
        if self.use_cross_attention and wind_features is not None:
            x = self.cross_attention(x, wind_features)
        
        return x


class DownBlock(nn.Module):
    """带有跨模态交叉注意力的下采样块"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, wind_dim, 
                 use_cross_attention=True):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # 基础卷积块
        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim)
        
        # 跨模态交叉注意力（可选，根据需求设置为False）
        if use_cross_attention:
            self.cross_attention = CrossModalCrossAttention(
                image_channels=out_channels,
                wind_channels=wind_dim,
                num_heads=8
            )
        
        # 下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, t, wind_features):
        # 卷积处理
        x = self.conv_block(x, t)
        
        # 跨模态注意力（如果启用）
        if self.use_cross_attention:
            x = self.cross_attention(x, wind_features)
        
        # 下采样
        p = self.pool(x)
        
        return x, p


class UpBlock(nn.Module):
    """带有跨模态交叉注意力的上采样块"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, wind_dim,
                 use_cross_attention=True, use_self_attention=False):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        
        # 上采样卷积
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        
        # 跨模态卷积块（注意：这里设置use_cross_attention=False，因为我们在输入处已经融合）
        self.conv_block = CrossModalConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            wind_dim=wind_dim,
            use_cross_attention=use_cross_attention
        )
        
        # 自注意力（可选）
        if use_self_attention:
            self.self_attention = AttentionBlock(out_channels)
    
    def forward(self, x, skip, t, wind_features):
        # 上采样
        x = self.upconv(x)
        
        # 确保尺寸匹配
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        
        # 拼接跳跃连接
        x = torch.cat((x, skip), dim=1)
        
        # 卷积处理 + 跨模态注意力
        x = self.conv_block(x, t, wind_features)
        
        # 自注意力（如果启用）
        if self.use_self_attention:
            x = self.self_attention(x)
        return x


class UNet(nn.Module):
    """
    修改后的跨模态UNet：太阳风向量条件极光生成
    
    主要修改：
    1. 在输入端添加跨模态交叉注意力，实现太阳风条件与图像的早期融合
    2. 取消采样路径中的太阳风特征融合（通过设置use_cross_attention=False）
    3. 保留瓶颈层的注意力机制
    """
    
    def __init__(self, 
                 in_channels=1,           # 输入通道数（图像）
                 out_channels=1,          # 输出通道数
                 time_emb_dim=256,        # 时间嵌入维度
                 solar_wind_dim=5,        # 太阳风参数维度
                 hidden_dim=128,          # 隐藏层维度
                 num_fourier_freq=10): # 瓶颈层使用注意力
        
        super().__init__()
        # self.num_fourier_freq = num_fourier_freq  # 保存参数
        # fourier_channels = 2 * num_fourier_freq
        
        # ==================== 太阳风编码器 ====================
        self.solar_wind_encoder = SolarWindEncoder(
            input_dim=solar_wind_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * 2
        )
        
        # ==================== 时间嵌入 ====================
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # ==================== 输入端融合模块 ====================
        # 新增：输入通道数调整，为交叉注意力做准备
        self.input_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        
        # 新增：输入端跨模态交叉注意力
        self.input_cross_attention = CrossModalCrossAttention(
            image_channels=32,          # 输入卷积后的通道数
            wind_channels=hidden_dim * 2,  # 太阳风编码后的维度
            num_heads=8
        )
        
        # ==================== 下采样路径 ====================
        # 第一层：输入通道数为32（输入卷积后的通道数）
        self.down1 = DownBlock(
            in_channels=32,  # 输入卷积后的通道数
            out_channels=64,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False  # 取消采样路径中的太阳风融合
        )
        
        self.down2 = DownBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False  # 取消采样路径中的太阳风融合
        )
        
        self.down3 = DownBlock(
            in_channels=128,
            out_channels=256,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False  # 取消采样路径中的太阳风融合
        )
        
        self.down4 = DownBlock(
            in_channels=256,
            out_channels=512,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False  # 取消采样路径中的太阳风融合
        )
        
        # ==================== 瓶颈层 ====================
        self.bottleneck = ConvBlock(
            in_channels=512,
            out_channels=1024,
            time_emb_dim=time_emb_dim,
        )
        
        self.self_attention_bottleneck = AttentionBlock(1024)
        
        # ==================== 上采样路径 ====================
        self.up1 = UpBlock(
            in_channels=1024,
            out_channels=512,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False
        )
        
        self.up2 = UpBlock(
            in_channels=512,
            out_channels=256,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False
        )
        
        self.up3 = UpBlock(
            in_channels=256,
            out_channels=128,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False
        )
        
        self.up4 = UpBlock(
            in_channels=128,
            out_channels=64,
            time_emb_dim=time_emb_dim,
            wind_dim=hidden_dim * 2,
            use_cross_attention=False
        )
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, timestep, solar_wind):
        """
        x: 噪声图像 [B, 1, H, W]
        timestep: 时间步 [B]
        solar_wind: 太阳风参数 [B, 5]
        """
        # ==================== 1. 编码太阳风参数 ====================
        wind_features = self.solar_wind_encoder(solar_wind)  # [B, hidden_dim*2]
        
        # ==================== 2. 输入端融合 ====================
        # 对输入图像进行通道调整
        #x_cond = self.add_fourier_features(x)  # 移除width参数
        x_cond = self.input_conv(x)  # [B, 32, H, W]
        
        # 输入端跨模态融合：图像特征与太阳风条件融合
        x_cond = self.input_cross_attention(x_cond, wind_features)  # [B, 32, H, W]
        
        # ==================== 3. 时间嵌入 ====================
        t = self.time_mlp(timestep)
        
        # ==================== 4. 下采样路径 ====================
        d1, p1 = self.down1(x_cond, t, wind_features)  # wind_features不再使用
        d2, p2 = self.down2(p1, t, wind_features)      # wind_features不再使用
        d3, p3 = self.down3(p2, t, wind_features)      # wind_features不再使用
        d4, p4 = self.down4(p3, t, wind_features)      # wind_features不再使用
        
        # ==================== 5. 瓶颈层 ====================
        b = self.bottleneck(p4, t) 
        #b = self.self_attention_bottleneck(b)
        
        # ==================== 6. 上采样路径 ====================
        u1 = self.up1(b, d4, t, wind_features)  # wind_features不再使用
        u2 = self.up2(u1, d3, t, wind_features)  # wind_features不再使用
        u3 = self.up3(u2, d2, t, wind_features)  # wind_features不再使用
        u4 = self.up4(u3, d1, t, wind_features)  # wind_features不再使用
        
        # ==================== 7. 最终输出 ====================
        output = self.final_conv(u4)
        
        return output