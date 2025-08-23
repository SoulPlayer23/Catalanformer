import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size_encoder: int,
        vocab_size_decoder: int,
        context_length_encoder: int,
        context_length_decoder: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size_encoder, context_length_encoder, model_dim, num_blocks, num_heads)
        self.decoder = Decoder(vocab_size_decoder, context_length_decoder, model_dim, num_blocks, num_heads)

    def forward(self, encoder_input_ids, decoder_input_ids, src_mask, tgt_mask):
        encoder_hidden_states = self.encoder(encoder_input_ids, src_mask)  
        decoder_logits = self.decoder(decoder_input_ids, encoder_hidden_states, src_mask, tgt_mask)

        return decoder_logits
