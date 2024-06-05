import torch
import torch.nn.functional as F
from torch import nn

from model.fine_tuning.batch_dense_cheb_conv import BatchDenseChebConv


# from torch_geometric.nn.conv import ChebConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, k_order, dropout_probs=None):
        super(GCN, self).__init__()

        if dropout_probs is None:
            dropout_probs = [0.5] * len(hidden_dims)
        if isinstance(dropout_probs, float):
            dropout_probs = [dropout_probs] * len(hidden_dims)

        # Add the input layer with dropout and activation
        layers = [
            BatchDenseChebConv(input_dim, hidden_dims[0], k_order),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dims[0]),  # TODO: Add back, but take note that channel dimension is constant
            # nn.Dropout(p=dropout_probs[0]),  # TODO: Add back
        ]

        # Add the hidden layers with dropout and activation
        for i in range(1, len(hidden_dims)):
            layers.extend([
                BatchDenseChebConv(hidden_dims[i - 1], hidden_dims[i], k_order),
                nn.ReLU(),
                # nn.BatchNorm1d(hidden_dims[i]),  # TODO: Add back
                # nn.Dropout(p=dropout_probs[i]),  # TODO: Add back
            ])

        # Combine all layers into a Sequential module
        self.gcn_layers = nn.Sequential(*layers)

    def forward(self, x, adj):
        for layer in self.gcn_layers:
            if isinstance(layer, BatchDenseChebConv):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x

    @staticmethod
    def build_adjacency_matrix(Z, delta=0.2):
        # TODO: Use inside class directly

        """
        Construct the adjacency matrix based on cosine similarity.

        Parameters:
            - Z (torch.Tensor): The node features matrix (BATCH_SIZE x V x F), where V is the number of channels
              and F is the feature dimension.
            - delta (float): The threshold value to determine the adjacency matrix entries.

        Returns:
            - adj_matrix (torch.Tensor): The constructed adjacency matrix (BATCH_SIZE x V x V).
        """
        Z_norm = F.normalize(Z, p=2, dim=-1)  # Normalize along the feature dimension
        sim_matrix = torch.bmm(Z_norm, Z_norm.transpose(-1, -2))  # Cosine similarity

        # Apply the adjacency matrix formula (have to use where so that it can be back-propagated)
        adj_matrix = torch.where(
            sim_matrix >= delta,
            torch.exp(sim_matrix - 1),
            torch.tensor(delta, device=Z.device),
        )

        return adj_matrix
