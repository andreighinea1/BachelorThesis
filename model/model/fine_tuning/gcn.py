import torch
from torch import nn
from torch.functional import F
from torch_geometric.nn.conv import ChebConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, k_order, dropout_probs=None):
        super(GCN, self).__init__()

        if dropout_probs is None:
            dropout_probs = [0.5] * len(hidden_dims)
        if isinstance(dropout_probs, float):
            dropout_probs = [dropout_probs] * len(hidden_dims)

        # Add the input layer with dropout and activation
        layers = [
            ChebConv(input_dim, hidden_dims[0], k_order),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(p=dropout_probs[0])
        ]

        # Add the hidden layers with dropout and activation
        for i in range(1, len(hidden_dims)):
            layers.extend([
                ChebConv(hidden_dims[i - 1], hidden_dims[i], k_order),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.Dropout(p=dropout_probs[i])
            ])

        # Combine all layers into a Sequential module
        self.gcn_layers = nn.Sequential(*layers)

    def forward(self, x, adj):
        for layer in self.gcn_layers:
            if isinstance(layer, ChebConv):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x

    @staticmethod
    def build_adjacency_matrix(Z, delta=0.2):
        # Z: (batch_size, V, F)
        Z_norm = F.normalize(Z, p=2, dim=-1)  # Normalize along the feature dimension
        sim_matrix = torch.matmul(Z_norm, Z_norm.transpose(-1, -2))  # Cosine similarity

        A = torch.where(
            sim_matrix >= delta,
            torch.exp(sim_matrix - 1),
            torch.tensor(delta).to(sim_matrix.device)  # Will be broadcasted to the right size
        )
        return A
