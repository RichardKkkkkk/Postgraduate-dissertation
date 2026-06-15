import torch
import torch.nn as nn


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def build_rope_cache_for_positions(positions, dim, device, dtype, base=10000):
    positions = positions.to(device=device, dtype=torch.float32)
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
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention2DRoPE(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        attention_dropout=0.0,
        projection_dropout=0.0,
        rope_base=10000,
        patch_grid_size=14,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_base = rope_base
        self.patch_grid_size = patch_grid_size
        self.patch_tokens = patch_grid_size * patch_grid_size
        self.axis_dim = self.head_dim // 2

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 when using 2D RoPE"

        row_positions = torch.arange(self.patch_tokens) // patch_grid_size
        col_positions = torch.arange(self.patch_tokens) % patch_grid_size
        self.register_buffer("row_positions", row_positions, persistent=False)
        self.register_buffer("col_positions", col_positions, persistent=False)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(projection_dropout)

    def apply_2d_rope(self, x):
        x_row, x_col = torch.chunk(x, 2, dim=-1)
        cos_row, sin_row = build_rope_cache_for_positions(
            positions=self.row_positions,
            dim=self.axis_dim,
            device=x.device,
            dtype=x.dtype,
            base=self.rope_base,
        )
        cos_col, sin_col = build_rope_cache_for_positions(
            positions=self.col_positions,
            dim=self.axis_dim,
            device=x.device,
            dtype=x.dtype,
            base=self.rope_base,
        )
        x_row = apply_rope(x_row, cos_row, sin_row)
        x_col = apply_rope(x_col, cos_col, sin_col)
        return torch.cat([x_row, x_col], dim=-1)

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if N > 1:
            if N - 1 != self.patch_tokens:
                raise ValueError(
                    f"Expected {self.patch_tokens} patch tokens for 2D RoPE, got {N - 1}"
                )
            rope_q = q[:, :, 1:, :]
            rope_k = k[:, :, 1:, :]
            q = torch.cat([q[:, :, :1, :], self.apply_2d_rope(rope_q)], dim=2)
            k = torch.cat([k[:, :, :1, :], self.apply_2d_rope(rope_k)], dim=2)

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


class Block2DRoPE(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        mlp_hidden_dim=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        rope_base=10000,
        patch_grid_size=14,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention2DRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            rope_base=rope_base,
            patch_grid_size=patch_grid_size,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=mlp_dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTRoPE2D(nn.Module):
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
        self.blocks = nn.ModuleList(
            [
                Block2DRoPE(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    mlp_dropout=mlp_dropout,
                    rope_base=rope_base,
                    patch_grid_size=self.patch_embed.grid_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)

    model = ViTRoPE2D(
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
