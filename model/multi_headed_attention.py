import torch
from torch import nn
from model.self_attention import SingleHeadAttention

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mask: bool = False):
                super().__init__()
                self.attention_heads = nn.ModuleList()
                for _ in range(num_heads):
                    self.attention_heads.append(SingleHeadAttention(model_dim, model_dim // num_heads, mask))
                self.compute = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout(0.2)

    def forward(self, query, key=None, value=None):
        """
        query: Tensor of shape (B, T_q, D)
        key:   Tensor of shape (B, T_k, D) or None (defaults to query)
        value: Tensor of shape (B, T_v, D) or None (defaults to key)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(query, key, value))

        concatenated = torch.cat(head_outputs, dim = -1)
        return self.dropout(self.compute(concatenated))