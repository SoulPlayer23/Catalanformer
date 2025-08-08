import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_size: int, mask: bool = True):
        super().__init__()
        self.key_layer = nn.Linear(model_dim, head_size, bias=False)
        self.query_layer = nn.Linear(model_dim, head_size, bias=False)
        self.value_layer = nn.Linear(model_dim, head_size, bias=False)
        self.mask = mask

    def forward(self, embedded):
        k = self.key_layer(embedded)
        q = self.query_layer(embedded)
        v = self.value_layer(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        if self.mask:
            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = (lower_triangular == 0).to(device)
            scores = scores.masked_fill(mask, float('-inf'))

        scores = nn.functional.softmax(scores, dim = 2)

        return scores @ v