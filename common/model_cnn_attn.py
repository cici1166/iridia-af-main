# common/model_cnn_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNNWithAttention", "count_params"]

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=None, pool=2):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool) if pool else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class AttentionPooling(nn.Module):
    """
    轻量时间注意力汇聚：score = tanh(Wx+b) -> softmax(time) -> 加权求和
    输入:  [B, T, C]
    输出:  [B, C]
    """
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.v    = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):  # x: [B, T, C]
        h = torch.tanh(self.proj(x))      # [B, T, A]
        score = self.v(h).squeeze(-1)     # [B, T]
        alpha = F.softmax(score, dim=1)   # [B, T]
        ctx = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)  # [B, C]
        return ctx, alpha

class CNNWithAttention(nn.Module):
    """
    单导联 ECG → CNN 形态特征 → AttentionPooling 汇聚 → FC → Logit
    - width_mult 调整通道规模（与其他模型参数量对齐）
    - fc_dim 全连接隐藏维度
    - attn_dim 注意力隐藏维度
    输入:  x: [B, 1, T]
    输出:  logits: [B, 1]
    """
    def __init__(self, width_mult: float = 1.0, fc_dim: int = 128, attn_dim: int = 128):
        super().__init__()
        def ch(c):  # 通道缩放
            return max(1, int(round(c * width_mult)))

        C1, C2, C3, C4 = ch(32), ch(64), ch(128), ch(128)

        self.backbone = nn.Sequential(
            ConvBlock(1,  C1, k=7, pool=4),
            ConvBlock(C1, C2, k=7, pool=4),
            ConvBlock(C2, C3, k=5, pool=4),
            ConvBlock(C3, C4, k=3, pool=2),
        )
        self.attn = AttentionPooling(in_dim=C4, attn_dim=attn_dim)

        self.head = nn.Sequential(
            nn.Linear(C4, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(fc_dim, 1)  # logits
        )

        # 默认最后一层 bias 为 0；如需缓解初期“全负/全正”，可由外部注入先验 bias。

    def forward(self, x):   # x: [B, 1, T]
        feat = self.backbone(x)          # [B, C4, T']
        feat_seq = feat.transpose(1, 2)  # [B, T', C4]
        ctx, _ = self.attn(feat_seq)     # [B, C4]
        logit = self.head(ctx)           # [B, 1]
        return logit
