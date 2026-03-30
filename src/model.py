"""
model.py — Multi-Scale Vision Transformer with Contrastive Learning (MViT-CnG)
This is the core model from the research paper.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 1: Patch Embedding
# Divides image into patches and converts each to an embedding vector
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly embeds them.
    
    Example: 48x48 image, patch_size=4
    → (48/4) × (48/4) = 144 patches
    → Each patch: 4×4×3 = 48 raw features → projected to embed_dim=768
    """
    
    def __init__(self, image_size=48, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # A convolution with stride=patch_size divides image into non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [batch, channels, height, width] = [B, 3, 48, 48]
        x = self.projection(x)          # [B, embed_dim, H/P, W/P] = [B, 768, 12, 12]
        x = x.flatten(2)               # [B, 768, 144]
        x = x.transpose(1, 2)          # [B, 144, 768] — (batch, sequence, features)
        return x
# BUILDING BLOCK 2: Multi-Head Self-Attention
# Each head learns different "aspects" of relationships
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    Computes attention between all patches simultaneously.
    12 heads × 64-dim = 768-dim total
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 768 / 12 = 64
        self.scale = self.head_dim ** -0.5      # Scaling factor (1/sqrt(64))
        
        # Projections for Query, Key, Value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape  # batch, num_patches, embed_dim
        
        # Compute Q, K, V in one operation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)           # Each: [B, heads, N, head_dim]
        
        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) × V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = torch.matmul(attn, v)         # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.out_proj(x)
        return x
 
# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 3: Feed-Forward Network (FFN)
# Applies a 2-layer MLP to each position independently
# ─────────────────────────────────────────────────────────────────────────────
class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear
    Expands dimension by 4x (768 → 3072 → 768)
    """
    
    def __init__(self, embed_dim=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)  # 768 × 4 = 3072
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()          # Smoother than ReLU, better for transformers
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
 
# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 4: Transformer Encoder Block
# One complete transformer layer = Attention + FFN with residual connections
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    One Transformer encoder block:
    x → LayerNorm → Attention → Add (residual) → LayerNorm → FFN → Add
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        # Residual connection 1: x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x))
        # Residual connection 2: x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        return x
 
# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 5: Single-Scale Vision Transformer
# Processes image at ONE specific scale
# ─────────────────────────────────────────────────────────────────────────────
class SingleScaleViT(nn.Module):
    """
    Vision Transformer for a single image scale.
    Image → Patches → Embeddings → Transformer Layers → CLS token
    """
    
    def __init__(self, image_size=48, patch_size=4, in_channels=3,
                 embed_dim=768, num_heads=12, num_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token: a learnable vector prepended to the sequence
        # Used to aggregate global information for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings: tells model the position of each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Convert image to patch embeddings
        x = self.patch_embed(x)         # [B, num_patches, embed_dim]
        
        # Prepend CLS token (duplicated for each sample in batch)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, num_patches+1, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return CLS token output (first token = global representation)
        return x[:, 0]  # [B, embed_dim]
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL: Multi-Scale Vision Transformer + Contrastive Learning
# ─────────────────────────────────────────────────────────────────────────────
class MViTCnG(nn.Module):
    """
    Multi-Scale Vision Transformer with Contrastive Learning (MViT-CnG)
    
    Architecture:
    1. Three ViT branches process the image at different patch sizes
    2. Features are concatenated → fused
    3. Contrastive projection head for NCE loss
    4. Classification head for emotion prediction
    """
    
    def __init__(self, num_classes=7, image_size=48, embed_dim=256,
                 num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.2,
                 contrastive_dim=128):
        super().__init__()
        
        # ── Scale 1: Small patches (fine-grained details) ──
        # patch_size=2: 48/2=24×24=576 patches — captures fine wrinkles, pores
        self.vit_scale1 = SingleScaleViT(
            image_size=image_size, patch_size=2, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # ── Scale 2: Medium patches ──
        # patch_size=4: 48/4=12×12=144 patches — captures mid-level features
        self.vit_scale2 = SingleScaleViT(
            image_size=image_size, patch_size=4, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # ── Scale 3: Large patches (coarse structure) ──
        # patch_size=8: 48/8=6×6=36 patches — captures overall face structure
        self.vit_scale3 = SingleScaleViT(
            image_size=image_size, patch_size=8, in_channels=3,
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # ── Feature Fusion: Combine 3 scale features ──
        fused_dim = embed_dim * 3  # 256 × 3 = 768
        
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )
        
        # ── Contrastive Projection Head ──
        # Projects features to a lower-dimensional space for NCE loss
        self.contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrastive_dim)  # 128-dim output
        )
        
        # ── Classification Head ──
        # Final emotion prediction
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: input image tensor [batch, 3, 48, 48]
        
        Returns:
            logits: class predictions [batch, num_classes]
            contrastive_features: for NCE loss [batch, 128]
        """
        # Process at 3 scales simultaneously
        f1 = self.vit_scale1(x)   # [B, embed_dim]
        f2 = self.vit_scale2(x)   # [B, embed_dim]
        f3 = self.vit_scale3(x)   # [B, embed_dim]
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([f1, f2, f3], dim=-1)  # [B, embed_dim × 3]
        
        # Fuse features
        fused = self.fusion(multi_scale)  # [B, embed_dim]
        
        # Contrastive features (for training only)
        contrastive_features = self.contrastive_head(fused)  # [B, 128]
        
        # Classification
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits, contrastive_features
    
    def predict(self, x):
        """
        For inference only — returns probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
 
 
def count_parameters(model):
    """Count trainable parameters — good to know model size."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total
