import torch
import torch.nn as nn

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
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, N, D = x.shape  # [B, N, D]

        qkv = self.qkv_proj(x)  # [B, N, 3*D]

        # qkv: [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = out.transpose(1, 2)
        out = out.reshape(B, N, D)
        out = self.out_proj(out)

        return out
    
class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_hidden_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):   
    def __init__(self,img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, mlp_hidden_dim=None, num_blocks=12, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)

        return logits



if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)

    model = ViT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        num_blocks=4,
        num_heads=4,
        mlp_hidden_dim=512
    )

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
