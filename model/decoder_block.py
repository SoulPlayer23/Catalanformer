import torch
from torch import nn
from model.multi_headed_attention import MultiHeadedAttention
from model.vanilla_neural_network import VanillaNeuralNetwork

class Decoder_Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.mhsa = MultiHeadedAttention(model_dim, num_heads, mask=True)
        self.cmhsa = MultiHeadedAttention(model_dim, num_heads, mask=False)  # Cross-attention
        self.vanilla_nn = VanillaNeuralNetwork(model_dim)
        self.layer_norm_one = nn.LayerNorm(model_dim)
        self.layer_norm_two = nn.LayerNorm(model_dim)
        self.layer_norm_three = nn.LayerNorm(model_dim)

    def forward(self, embedded, encoder_output):
        embedded = embedded + self.mhsa(query=self.layer_norm_one(embedded), key=embedded, value=embedded) # skip connection
        embedded = embedded + self.cmhsa(query=self.layer_norm_two(embedded), key=encoder_output, value=encoder_output) # cross attention skip connection
        embedded = embedded + self.vanilla_nn(self.layer_norm_three(embedded)) # another skip connection
        return embedded