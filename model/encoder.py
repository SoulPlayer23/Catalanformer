import torch
from torch import nn
from model.encoder_block import Encoder_Block

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        self.transformer_blocks = nn.ModuleList(
            [Encoder_Block(model_dim, num_heads) for _ in range(num_blocks)]
        )
        self.layer_norm_three = nn.LayerNorm(model_dim)
    
    def forward(self, context, src_mask):
        embedded = self.token_embedding(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length, device=context.device)
        pos_embedding = self.pos_embedding(positions).unsqueeze(0)
        embedded = embedded + pos_embedding

        for block in self.transformer_blocks:
            embedded = block(embedded, src_mask)

        hidden_states = self.layer_norm_three(embedded)  # (B, T, model_dim)

        return hidden_states