import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, k_order):
        super(GCN, self).__init__()
        self.gcn1 = ChebConv(input_dim, hidden_dim1, k_order)
        self.gcn2 = ChebConv(hidden_dim1, hidden_dim2, k_order)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x
