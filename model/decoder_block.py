import torch
from torch import nn
from model.vanilla_neural_network import VanillaNeuralNetwork

class Decoder_Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.masked_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.vanilla_nn = VanillaNeuralNetwork(model_dim)
        self.layer_norm_one = nn.LayerNorm(model_dim)
        self.layer_norm_two = nn.LayerNorm(model_dim)
        self.layer_norm_three = nn.LayerNorm(model_dim)

    def forward(self, embedded, encoder_output, src_mask, tgt_mask):
        if src_mask is not None:
            src_mask = ~src_mask.squeeze(1).squeeze(1)
        
        tgt_padding_mask = ~tgt_mask.squeeze(1)[:, 0, :]
        tgt_len = embedded.size(1)
        look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=embedded.device), diagonal=1).bool()
        
        masked_attn_output, _ = self.masked_attention(
            query=embedded, key=embedded, value=embedded,
            key_padding_mask=tgt_padding_mask,
            attn_mask=look_ahead_mask,
            need_weights=False
        )
        add_1 = embedded + masked_attn_output
        norm_1 = self.layer_norm_one(add_1)

        cross_attn_output, _ = self.cross_attention(
            query=norm_1, key=encoder_output, value=encoder_output,
            key_padding_mask=src_mask,
            need_weights=False
        )
        add_2 = norm_1 + cross_attn_output
        norm_2 = self.layer_norm_two(add_2)

        ff_output = self.vanilla_nn(norm_2)
        add_3 = norm_2 + ff_output
        norm_3 = self.layer_norm_three(add_3)
        
        return norm_3
