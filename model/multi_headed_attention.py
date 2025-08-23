import torch
from torch import nn
from model.self_attention import SingleHeadAttention

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        head_size = model_dim // num_heads
        self.attention_heads = nn.ModuleList(
            [SingleHeadAttention(model_dim, head_size) for _ in range(num_heads)]
        )
        self.compute = nn.Linear(model_dim, model_dim) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(query, key, value, mask))

        # Concatenate along the last dimension (the feature dimension)
        concatenated = torch.cat(head_outputs, dim=-1)
        
        return self.dropout(self.compute(concatenated))
