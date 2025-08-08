import torch
from torch import nn
from model.multi_headed_attention import MultiHeadedAttention
from model.vanilla_neural_network import VanillaNeuralNetwork

class Encoder_Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.mhsa = MultiHeadedAttention(model_dim, num_heads, mask=False)
        self.vanilla_nn = VanillaNeuralNetwork(model_dim)
        self.layer_norm_one = nn.LayerNorm(model_dim)
        self.layer_norm_two = nn.LayerNorm(model_dim)

    def forward(self, embedded):
        embedded = embedded + self.mhsa(self.layer_norm_one(embedded)) # skip connection
        embedded = embedded + self.vanilla_nn(self.layer_norm_two(embedded)) # another skip connection
        return embedded