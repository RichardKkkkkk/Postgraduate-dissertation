import math

import torch
import torch.nn as nn

from .vit import Block, PatchEmbedding


def build_sinusoidal_embedding(positions, embed_dim):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even when using sinusoidal positional embeddings")

    positions = positions.to(dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )
    embedding = torch.zeros((positions.shape[0], embed_dim), dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(positions * div_term)
    embedding[:, 1::2] = torch.cos(positions * div_term)
    return embedding


def build_axis_positions(grid_size, axis):
    token_positions = torch.arange(grid_size * grid_size)
    if axis == "row":
        return token_positions // grid_size
    if axis == "col":
        return token_positions % grid_size
    raise ValueError(f"Unsupported axis variant: {axis}")


class ViTAxisSinusoidal(nn.Module):
    def __init__(
        self,
        axis,
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
    ):
        super().__init__()
        self.axis = axis
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(embedding_dropout)

        axis_positions = build_axis_positions(self.patch_embed.img_size // self.patch_embed.patch_size, axis)
        patch_pos_embed = build_sinusoidal_embedding(axis_positions, embed_dim)
        cls_pos_embed = torch.zeros((1, embed_dim), dtype=torch.float32)
        full_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=0).unsqueeze(0)
        self.register_buffer("pos_embed", full_pos_embed, persistent=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_hidden_dim,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    mlp_dropout=mlp_dropout,
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
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


class ViTRowSinusoidal(ViTAxisSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(axis="row", **kwargs)


class ViTColSinusoidal(ViTAxisSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(axis="col", **kwargs)


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)

    row_model = ViTRowSinusoidal(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=2,
        embed_dim=128,
        num_blocks=4,
        num_heads=4,
        mlp_hidden_dim=512,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        projection_dropout=0.1,
        mlp_dropout=0.1,
    )
    col_model = ViTColSinusoidal(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=2,
        embed_dim=128,
        num_blocks=4,
        num_heads=4,
        mlp_hidden_dim=512,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        projection_dropout=0.1,
        mlp_dropout=0.1,
    )

    print("Row logits shape:", row_model(x).shape)
    print("Col logits shape:", col_model(x).shape)
