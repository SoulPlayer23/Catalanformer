import torch
from torch import nn
from model.vanilla_neural_network import VanillaNeuralNetwork

class Encoder_Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.vanilla_nn = VanillaNeuralNetwork(model_dim)
        self.layer_norm_one = nn.LayerNorm(model_dim)
        self.layer_norm_two = nn.LayerNorm(model_dim)

    def forward(self, embedded, src_mask):
        if src_mask is not None:
            src_mask = ~src_mask.squeeze(1).squeeze(1)

        attention_output, _ = self.attention(
            query=embedded, 
            key=embedded, 
            value=embedded, 
            key_padding_mask=src_mask,
            need_weights=False
        )
        
        add_1 = embedded + attention_output
        norm_1 = self.layer_norm_one(add_1)

        ff_output = self.vanilla_nn(norm_1)
        add_2 = norm_1 + ff_output
        norm_2 = self.layer_norm_two(add_2)
        
        return norm_2
