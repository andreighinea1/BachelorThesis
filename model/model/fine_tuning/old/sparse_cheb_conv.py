import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class SparseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(SparseChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.out_channels)
        self.weights.data.uniform_(-stdv, stdv)

    def _rescale_laplacian(self, laplacian):
        laplacian = sp.csr_matrix(laplacian)
        I = sp.identity(laplacian.shape[0], format='csr')
        L = laplacian - I
        lambda_max = sp.linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
        L = (2 / lambda_max) * L - I
        return L

    def forward(self, x, adj):
        batch_size, num_nodes, in_channels = x.size()
        assert in_channels == self.in_channels

        # Create Laplacian for each graph in the batch
        I = sp.eye(num_nodes).tocoo()
        indices = torch.tensor(np.column_stack((I.row, I.col)), dtype=torch.long)
        I = torch.sparse.FloatTensor(indices.t(), torch.tensor(I.data, dtype=torch.float32), torch.Size(I.shape)).to(
            x.device)

        adj_normalized = self._normalize_adj(adj)
        L = I - adj_normalized
        L = self._rescale_laplacian(L)

        # Convert L to sparse tensor
        L = L.tocoo()
        indices = torch.tensor(np.column_stack((L.row, L.col)), dtype=torch.long)
        L = torch.sparse.FloatTensor(indices.t(), torch.tensor(L.data, dtype=torch.float32), torch.Size(L.shape)).to(
            x.device)

        out = torch.zeros(batch_size, num_nodes, self.out_channels).to(x.device)
        Tx_0 = x
        Tx_1 = torch.bmm(L.to_dense(), x)
        out += torch.matmul(Tx_0, self.weights[0])
        if self.K > 1:
            out += torch.matmul(Tx_1, self.weights[1])
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(L.to_dense(), Tx_1) - Tx_0
            out += torch.matmul(Tx_2, self.weights[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def _normalize_adj(self, adj):
        # Normalize adjacency matrix
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
