import torch
from torch import nn
from model.decoder_block import Decoder_Block

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.ModuleList(
            [Decoder_Block(model_dim, num_heads) for _ in range(num_blocks)]
        )
        self.layer_norm_final = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, decoder_input_ids, encoder_output, src_mask, tgt_mask):
        embedded = self.token_embedding(decoder_input_ids)
        context_length = decoder_input_ids.shape[1]
        positions = torch.arange(context_length, device=decoder_input_ids.device)
        pos_embedding = self.pos_embedding(positions).unsqueeze(0)
        embedded = embedded + pos_embedding

        for block in self.transformer_blocks:
            embedded = block(embedded, encoder_output, src_mask, tgt_mask)

        logits = self.vocab_projection(self.layer_norm_final(embedded))
        # logits is BxTxV, where V is the vocabulary size
        return logits