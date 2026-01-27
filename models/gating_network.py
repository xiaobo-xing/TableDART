import torch
import torch.nn as nn


class MlpGatingNetwork(nn.Module):
    """
    Lightweight MLP-based gating network for dynamic expert routing.
    Selects the optimal path (Text-only, Image-only, or Fusion) for each table-query pair.
    """
    def __init__(self, input_dim, num_paths, hidden_dim=256):
        super().__init__()
        self.num_paths = num_paths
        # Two-layer MLP with hidden layer and ReLU activation
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # Output layer produces logits for each routing path
        self.layer_2 = nn.Linear(hidden_dim, num_paths)
        print(
            f"INFO: Initialized MLP Gating Network (Input: {input_dim}, Hidden: {hidden_dim}, Output: {num_paths} paths)"
        )

    def forward(self, x):
        # Pass through first layer with activation
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Output routing logits
        logits = self.layer_2(x)
        return logits
