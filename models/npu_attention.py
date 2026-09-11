# models/npu_attention.py
"""
NPU 兼容的多头注意力实现
避免使用 nn.MultiheadAttention 触发 CPU fallback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NPUMultiheadAttention(nn.Module):
    """
    手动实现的多头注意力，完全在 NPU 上运行
    兼容 nn.MultiheadAttention 的接口
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V 投影（合并成一个矩阵加速）
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """
        Args:
            query: [B, T, D] if batch_first else [T, B, D]
            key: [B, S, D] if batch_first else [S, B, D]
            value: [B, S, D] if batch_first else [S, B, D]
            attn_mask: [T, S] 或 [B*num_heads, T, S]
            key_padding_mask: [B, S]
        Returns:
            attn_output: [B, T, D] if batch_first else [T, B, D]
            attn_weights: [B, num_heads, T, S] (如果 need_weights=True)
        """
        # 处理输入格式
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, T, D = query.shape
        S = key.shape[1]
        
        # === Self-Attention 优化路径（Q=K=V）===
        if query is key and key is value:
            # 合并 QKV 投影（3倍加速）
            qkv = self.qkv_proj(query)  # [B, T, 3*D]
            qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, d]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # 分开投影（Cross-Attention）
            q = self.qkv_proj(query)[:, :, :self.embed_dim]
            k = self.qkv_proj(key)[:, :, :self.embed_dim]
            v = self.qkv_proj(value)[:, :, :self.embed_dim]
            
            q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
            k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
            v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
        
        # === 计算注意力分数 ===
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, S]
        
        # 应用 attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # [1, T, S]
            attn_scores = attn_scores + attn_mask
        
        # 应用 key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, S]
                float('-inf')
            )
        
        # Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, d]
        
        # 重组多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, H, d]
        attn_output = attn_output.reshape(B, T, D)  # [B, T, D]
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # 恢复输入格式
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        if need_weights:
            return attn_output, attn_weights.mean(dim=1)  # 平均所有 head
        else:
            return attn_output, None


class NPUMultiheadAttentionSimple(nn.Module):
    """
    简化版单头注意力（用于池化等场景）
    性能更优，适合 num_heads=1 的情况
    """
    def __init__(self, embed_dim, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_first = batch_first
        self.scale = 1.0 / math.sqrt(embed_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # 投影
        q = self.q_proj(query)  # [B, T, D]
        k = self.k_proj(key)    # [B, S, D]
        v = self.v_proj(value)  # [B, S, D]
        
        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, S]
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),  # [B, 1, S]
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [B, T, D]
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, None