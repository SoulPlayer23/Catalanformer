import torch
from torch import nn

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))