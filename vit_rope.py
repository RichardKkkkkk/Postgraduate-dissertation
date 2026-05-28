import torch
import torch.nn as nn


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def build_rope_cache(seq_len, dim, device, dtype, base=10000):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    angles = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    cos = angles.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = angles.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        attention_dropout=0.0,
        projection_dropout=0.0,
        rope_base=10000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_base = rope_base

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even when using RoPE"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, D = x.shape  # [B, N, D]

        qkv = self.qkv_proj(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        if N > 1:
            rope_q = q[:, :, 1:, :]
            rope_k = k[:, :, 1:, :]
            cos, sin = build_rope_cache(
                seq_len=N - 1,
                dim=self.head_dim,
                device=x.device,
                dtype=x.dtype,
                base=self.rope_base,
            )
            q = torch.cat([q[:, :, :1, :], apply_rope(rope_q, cos, sin)], dim=2)
            k = torch.cat([k[:, :, :1, :], apply_rope(rope_k, cos, sin)], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(B, N, D)
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
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


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        mlp_hidden_dim=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        rope_base=10000,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=mlp_dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTRoPE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        mlp_hidden_dim=None,
        num_blocks=12,
        num_classes=10,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        rope_base=10000,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_hidden_dim,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
                mlp_dropout=mlp_dropout,
                rope_base=rope_base,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)

        return logits


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)

    model = ViTRoPE(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        num_blocks=4,
        num_heads=4,
        mlp_hidden_dim=512,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        projection_dropout=0.1,
        mlp_dropout=0.1,
    )

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
