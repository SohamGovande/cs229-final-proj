import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads)
        self.mlp = MLP(config.hidden_dim, config.mlp_dim, config.dropout_rate)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.attn_weights = None

    def forward(self, x):
        n_x = self.norm1(x)
        attn_output, self.attn_weights = self.attention(n_x, n_x, n_x)
        x = x + attn_output
        return x + self.mlp(self.norm2(x))

class TransformerBlockStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

    def forward(self, x):
        attn_weights = []
        for layer in self.layers:
            x = layer(x)
            attn_weights.append(layer.attn_weights)
        return x, attn_weights
