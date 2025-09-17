import torch
import torch.nn as nn


class MlpGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_paths, hidden_dim=256):
        super().__init__()
        self.num_paths = num_paths
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(hidden_dim, num_paths)
        print(
            f"INFO: Initialized MLP Gating Network (Input: {input_dim}, Hidden: {hidden_dim}, Output: {num_paths} paths)"
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.layer_2(x)
        return logits
