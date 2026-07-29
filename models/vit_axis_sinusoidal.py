import math

import torch
import torch.nn as nn

from .vit import Block, MLP, PatchEmbedding


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


def build_grid_positions(grid_size):
    token_positions = torch.arange(grid_size * grid_size)
    row_positions = token_positions // grid_size
    col_positions = token_positions % grid_size
    return row_positions, col_positions


def build_additive_2d_embedding(row_positions, col_positions, embed_dim):
    row_embed = build_sinusoidal_embedding(row_positions, embed_dim)
    col_embed = build_sinusoidal_embedding(col_positions, embed_dim)
    return row_embed + col_embed


def build_multiplicative_2d_embedding(row_positions, col_positions, embed_dim):
    row_embed = build_sinusoidal_embedding(row_positions, embed_dim)
    col_embed = build_sinusoidal_embedding(col_positions, embed_dim)
    return row_embed * col_embed


def build_radial_2d_embedding(row_positions, col_positions, embed_dim):
    row_positions = row_positions.to(dtype=torch.float32)
    col_positions = col_positions.to(dtype=torch.float32)
    radial_positions = torch.sqrt(row_positions.pow(2) + col_positions.pow(2))
    return build_sinusoidal_embedding(radial_positions, embed_dim)


def build_squared_multiplicative_2d_embedding(row_positions, col_positions, embed_dim):
    return build_multiplicative_2d_embedding(row_positions, col_positions, embed_dim).pow(2)


def build_shifted_sinusoidal_components(row_positions, col_positions, embed_dim):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even when using sinusoidal positional embeddings")

    row_positions = row_positions.to(dtype=torch.float32).unsqueeze(1)
    col_positions = col_positions.to(dtype=torch.float32).unsqueeze(1)

    row_div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )
    col_div_term = torch.exp(
        torch.arange(1, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )

    row_embed = torch.zeros((row_positions.shape[0], embed_dim), dtype=torch.float32)
    col_embed = torch.zeros((col_positions.shape[0], embed_dim), dtype=torch.float32)

    row_embed[:, 0::2] = torch.sin(row_positions * row_div_term)
    row_embed[:, 1::2] = torch.cos(row_positions * row_div_term)
    col_embed[:, 0::2] = torch.sin(col_positions * col_div_term)
    col_embed[:, 1::2] = torch.cos(col_positions * col_div_term)
    return row_embed, col_embed


def build_additive_shifted_2d_embedding(row_positions, col_positions, embed_dim):
    row_embed, col_embed = build_shifted_sinusoidal_components(row_positions, col_positions, embed_dim)
    return row_embed + col_embed


def build_multiplicative_shifted_2d_embedding(row_positions, col_positions, embed_dim):
    row_embed, col_embed = build_shifted_sinusoidal_components(row_positions, col_positions, embed_dim)
    return row_embed * col_embed


def build_squared_multiplicative_shifted_2d_embedding(row_positions, col_positions, embed_dim):
    return build_multiplicative_shifted_2d_embedding(row_positions, col_positions, embed_dim).pow(2)


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
        unfolding="normal_row",
    ):
        super().__init__()
        self.axis = axis
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, unfolding=unfolding)
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

    def forward_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_features(self, x):
        tokens = self.forward_tokens(x)
        cls_output = tokens[:, 0]
        return cls_output

    def forward(self, x):
        cls_output = self.forward_features(x)
        logits = self.head(cls_output)
        return logits


class ViTRowSinusoidal(ViTAxisSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(axis="row", **kwargs)


class ViTColSinusoidal(ViTAxisSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(axis="col", **kwargs)


class ViTRadialSinusoidal(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, unfolding=unfolding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(embedding_dropout)

        grid_size = self.patch_embed.img_size // self.patch_embed.patch_size
        row_positions, col_positions = build_grid_positions(grid_size)
        patch_pos_embed = build_radial_2d_embedding(row_positions, col_positions, embed_dim)
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


class ViTAdditiveSinusoidal(nn.Module):
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
        shifted_wavelength=False,
        unfolding="normal_row",
    ):
        super().__init__()
        self.shifted_wavelength = shifted_wavelength
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, unfolding=unfolding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(embedding_dropout)

        grid_size = self.patch_embed.img_size // self.patch_embed.patch_size
        row_positions, col_positions = build_grid_positions(grid_size)
        if shifted_wavelength:
            patch_pos_embed = build_additive_shifted_2d_embedding(row_positions, col_positions, embed_dim)
        else:
            patch_pos_embed = build_additive_2d_embedding(row_positions, col_positions, embed_dim)
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


class ViTAdditiveSinusoidalShifted(ViTAdditiveSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(shifted_wavelength=True, **kwargs)


class ViTMultiplicativeSinusoidal(nn.Module):
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
        shifted_wavelength=False,
        squared=False,
        unfolding="normal_row",
    ):
        super().__init__()
        self.shifted_wavelength = shifted_wavelength
        self.squared = squared
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, unfolding=unfolding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(embedding_dropout)

        grid_size = self.patch_embed.img_size // self.patch_embed.patch_size
        row_positions, col_positions = build_grid_positions(grid_size)
        if shifted_wavelength and squared:
            patch_pos_embed = build_squared_multiplicative_shifted_2d_embedding(
                row_positions, col_positions, embed_dim
            )
        elif shifted_wavelength:
            patch_pos_embed = build_multiplicative_shifted_2d_embedding(row_positions, col_positions, embed_dim)
        elif squared:
            patch_pos_embed = build_squared_multiplicative_2d_embedding(row_positions, col_positions, embed_dim)
        else:
            patch_pos_embed = build_multiplicative_2d_embedding(row_positions, col_positions, embed_dim)
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


class ViTMultiplicativeSinusoidalShifted(ViTMultiplicativeSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(shifted_wavelength=True, **kwargs)


class ViTSquaredMultiplicativeSinusoidal(ViTMultiplicativeSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(squared=True, **kwargs)


class ViTSquaredMultiplicativeSinusoidalShifted(ViTMultiplicativeSinusoidal):
    def __init__(self, **kwargs):
        super().__init__(shifted_wavelength=True, squared=True, **kwargs)


class ViTLearnableMultiplicativeSinusoidal(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, unfolding=unfolding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.fixed_pos_scale = nn.Parameter(torch.zeros(1))
        self.pos_dropout = nn.Dropout(embedding_dropout)

        grid_size = self.patch_embed.img_size // self.patch_embed.patch_size
        row_positions, col_positions = build_grid_positions(grid_size)
        patch_pos_embed = build_multiplicative_2d_embedding(row_positions, col_positions, embed_dim)
        cls_pos_embed = torch.zeros((1, embed_dim), dtype=torch.float32)
        full_fixed_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=0).unsqueeze(0)
        self.register_buffer("fixed_pos_embed", full_fixed_pos_embed, persistent=False)

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
        fixed_pos_embed = self.fixed_pos_embed.to(dtype=x.dtype, device=x.device)
        x = x + self.pos_embed + self.fixed_pos_scale.to(dtype=x.dtype) * fixed_pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


class ViTRowColLatentFusion(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        fusion_hidden_dim = mlp_hidden_dim or embed_dim * 4
        encoder_kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "mlp_hidden_dim": mlp_hidden_dim,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "embedding_dropout": embedding_dropout,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "mlp_dropout": mlp_dropout,
            "unfolding": unfolding,
        }

        self.row_encoder = ViTAxisSinusoidal(axis="row", **encoder_kwargs)
        self.col_encoder = ViTAxisSinusoidal(axis="col", **encoder_kwargs)
        self.row_encoder.head = nn.Identity()
        self.col_encoder.head = nn.Identity()

        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.Dropout(projection_dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        row_latent = self.row_encoder.forward_features(x)
        col_latent = self.col_encoder.forward_features(x)
        fused_latent = torch.cat([row_latent, col_latent], dim=1)
        fused_latent = self.fusion(fused_latent)
        logits = self.head(fused_latent)
        return logits


class ViTRowColMeanFusion(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        encoder_kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "mlp_hidden_dim": mlp_hidden_dim,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "embedding_dropout": embedding_dropout,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "mlp_dropout": mlp_dropout,
            "unfolding": unfolding,
        }

        self.row_encoder = ViTAxisSinusoidal(axis="row", **encoder_kwargs)
        self.col_encoder = ViTAxisSinusoidal(axis="col", **encoder_kwargs)
        self.row_encoder.head = nn.Identity()
        self.col_encoder.head = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        row_latent = self.row_encoder.forward_features(x)
        col_latent = self.col_encoder.forward_features(x)
        fused_latent = (row_latent + col_latent) / 2
        logits = self.head(fused_latent)
        return logits


class ViTRowColMeanMLPFusion(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        fusion_hidden_dim = mlp_hidden_dim or embed_dim * 4
        encoder_kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "mlp_hidden_dim": mlp_hidden_dim,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "embedding_dropout": embedding_dropout,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "mlp_dropout": mlp_dropout,
            "unfolding": unfolding,
        }

        self.row_encoder = ViTAxisSinusoidal(axis="row", **encoder_kwargs)
        self.col_encoder = ViTAxisSinusoidal(axis="col", **encoder_kwargs)
        self.row_encoder.head = nn.Identity()
        self.col_encoder.head = nn.Identity()
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.Dropout(projection_dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        row_latent = self.row_encoder.forward_features(x)
        col_latent = self.col_encoder.forward_features(x)
        fused_latent = (row_latent + col_latent) / 2
        fused_latent = self.fusion(fused_latent)
        logits = self.head(fused_latent)
        return logits


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(projection_dropout)

    def forward(self, query_tokens, context_tokens):
        B, N, D = query_tokens.shape
        _, M, _ = context_tokens.shape

        q = self.q_proj(query_tokens)
        kv = self.kv_proj(context_tokens)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        kv = kv.reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(B, N, D)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        mlp_hidden_dim=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim,
            num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=mlp_dropout)

    def forward(self, query_tokens, context_tokens):
        query_tokens = query_tokens + self.cross_attn(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
        )
        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class ViTRowColCrossAttentionFusion(nn.Module):
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
        unfolding="normal_row",
    ):
        super().__init__()
        encoder_kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "mlp_hidden_dim": mlp_hidden_dim,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "embedding_dropout": embedding_dropout,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "mlp_dropout": mlp_dropout,
            "unfolding": unfolding,
        }

        self.row_encoder = ViTAxisSinusoidal(axis="row", **encoder_kwargs)
        self.col_encoder = ViTAxisSinusoidal(axis="col", **encoder_kwargs)
        self.row_encoder.head = nn.Identity()
        self.col_encoder.head = nn.Identity()
        self.row_to_col = CrossAttentionBlock(
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
        )
        self.col_to_row = CrossAttentionBlock(
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
        )
        self.head = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x):
        row_tokens = self.row_encoder.forward_tokens(x)
        col_tokens = self.col_encoder.forward_tokens(x)

        row_cross_tokens = self.row_to_col(row_tokens, col_tokens)
        col_cross_tokens = self.col_to_row(col_tokens, row_tokens)

        row_cls = row_cross_tokens[:, 0]
        col_cls = col_cross_tokens[:, 0]
        fused_latent = torch.cat([row_cls, col_cls], dim=1)
        logits = self.head(fused_latent)
        return logits


class ViTRowColCrossAttentionMLPHeadFusion(ViTRowColCrossAttentionFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.head.in_features // 2
        num_classes = self.head.out_features
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )


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

    additive_model = ViTAdditiveSinusoidal(
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
    additive_shifted_model = ViTAdditiveSinusoidalShifted(
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
    multiplicative_model = ViTMultiplicativeSinusoidal(
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
    multiplicative_shifted_model = ViTMultiplicativeSinusoidalShifted(
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
    squared_multiplicative_model = ViTSquaredMultiplicativeSinusoidal(
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
    squared_multiplicative_shifted_model = ViTSquaredMultiplicativeSinusoidalShifted(
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
    radial_model = ViTRadialSinusoidal(
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
    learnable_multiplicative_model = ViTLearnableMultiplicativeSinusoidal(
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
        unfolding="normal_col",
    )
    row_col_fusion_model = ViTRowColLatentFusion(
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
    row_col_mean_fusion_model = ViTRowColMeanFusion(
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
    row_col_mean_mlp_fusion_model = ViTRowColMeanMLPFusion(
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
    row_col_cross_attention_fusion_model = ViTRowColCrossAttentionFusion(
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
    row_col_cross_attention_mlp_head_fusion_model = ViTRowColCrossAttentionMLPHeadFusion(
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

    print("Additive logits shape:", additive_model(x).shape)
    print("Additive shifted logits shape:", additive_shifted_model(x).shape)
    print("Multiplicative logits shape:", multiplicative_model(x).shape)
    print("Multiplicative shifted logits shape:", multiplicative_shifted_model(x).shape)
    print("Squared multiplicative logits shape:", squared_multiplicative_model(x).shape)
    print("Squared multiplicative shifted logits shape:", squared_multiplicative_shifted_model(x).shape)
    print("Radial logits shape:", radial_model(x).shape)
    print("Learnable + multiplicative logits shape:", learnable_multiplicative_model(x).shape)
    print("Row/column latent fusion logits shape:", row_col_fusion_model(x).shape)
    print("Row/column mean fusion logits shape:", row_col_mean_fusion_model(x).shape)
    print("Row/column mean MLP fusion logits shape:", row_col_mean_mlp_fusion_model(x).shape)
    print("Row/column cross-attention fusion logits shape:", row_col_cross_attention_fusion_model(x).shape)
    print(
        "Row/column cross-attention MLP-head fusion logits shape:",
        row_col_cross_attention_mlp_head_fusion_model(x).shape,
    )
