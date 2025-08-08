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

    def forward(self, embedded):
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(embedded))
        concatenated = torch.cat(head_outputs, dim = 2)
        return self.dropout(self.compute(concatenated))