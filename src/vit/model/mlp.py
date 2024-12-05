import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = F.gelu(self.dense(x))
        return self.out(self.dropout(x))
