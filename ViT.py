import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from PatchEmbed import PatchEmbed
from componentsX import *

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_channels = 3, num_classes = 1000,
                 embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4.0,
                 attn_drop_rate = 0.0, drop_rate = 0.0, norm_layer = nn.LayerNorm):
        
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, attn_drop_rate, drop_rate) for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std = 0.02)
        nn.init.constant_(self.cls_token, 0)
        nn.init.normal_(self.head.weight, std = 0.02)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

model = VisionTransformer()
print(model)
