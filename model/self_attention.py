import torch
from torch import nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key_layer = nn.Linear(model_dim, head_size, bias=False)
        self.query_layer = nn.Linear(model_dim, head_size, bias=False)
        self.value_layer = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, query, key, value, mask=None):
        # All inputs are (B, T, model_dim)
        k = self.key_layer(key)
        q = self.query_layer(query)
        v = self.value_layer(value)

        # scores shape: (B, T_q, T_k)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        if mask is not None:
            # Ensure mask is broadcastable
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = nn.functional.softmax(scores, dim=-1)

        # output shape: (B, T_q, head_size)
        return attention_weights @ v
