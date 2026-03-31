"""
model.py — Multi-Scale Vision Transformer with Contrastive Learning (MViT-CnG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=48, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection  = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)   # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)         # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)    # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim=256, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn   = FeedForwardNetwork(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SingleScaleViT(nn.Module):
    def __init__(self, image_size=48, patch_size=4, in_channels=3,
                 embed_dim=768, num_heads=12, num_layers=4,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.image_size  = image_size
        self.patch_size  = patch_size
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout   = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Always resize to expected image size — fixes shape mismatch for any input
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False
            )

        x = self.patch_embed(x)                        # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, num_patches+1, embed_dim]

        # Handle positional embedding size mismatch safely
        if x.shape[1] != self.pos_embed.shape[1]:
            cls_pos    = self.pos_embed[:, :1, :]
            patch_pos  = self.pos_embed[:, 1:, :]
            h = w      = int(patch_pos.shape[1] ** 0.5)
            patch_pos  = patch_pos.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            new_size   = int((x.shape[1] - 1) ** 0.5)
            patch_pos  = F.interpolate(
                patch_pos, size=(new_size, new_size),
                mode='bicubic', align_corners=False
            )
            patch_pos  = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            pos_embed  = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # return CLS token


class MViTCnG(nn.Module):
    def __init__(self, num_classes=7, image_size=48, embed_dim=256,
                 num_heads=8, num_layers=4, mlp_ratio=4,
                 dropout=0.2, contrastive_dim=128):
        super().__init__()

        # Three ViT branches at different patch sizes
        self.vit_scale1 = SingleScaleViT(
            image_size=image_size, patch_size=2, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.vit_scale2 = SingleScaleViT(
            image_size=image_size, patch_size=4, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.vit_scale3 = SingleScaleViT(
            image_size=image_size, patch_size=8, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )

        fused_dim = embed_dim * 3

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrastive_dim)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        f1 = self.vit_scale1(x)
        f2 = self.vit_scale2(x)
        f3 = self.vit_scale3(x)

        multi_scale = torch.cat([f1, f2, f3], dim=-1)
        fused       = self.fusion(multi_scale)

        contrastive_features = self.contrastive_head(fused)
        logits               = self.classifier(fused)

        return logits, contrastive_features

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total
