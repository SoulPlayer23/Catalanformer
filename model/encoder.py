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
        for _ in range(num_blocks):
            self.transformer_blocks.append(Encoder_Block(model_dim, num_heads))
        self.layer_norm_three = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)
    
    def forward(self, context):
        embedded = self.token_embedding(context)
        positions = torch.arange(context.shape[1], device=embedded.device)
        embedded = embedded + self.pos_embedding(positions)

        hidden_states = self.layer_norm_three(self.transformer_blocks(embedded))  # (B, T, model_dim)
        logits = self.vocab_projection(hidden_states)  # (B, T, vocab_size)
        # logits is BxTxV, where V is the vocabulary size

        return hidden_states, logits