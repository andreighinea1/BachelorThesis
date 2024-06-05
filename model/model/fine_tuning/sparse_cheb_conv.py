import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class SparseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, laplacian):
        super(SparseChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.laplacian = self._rescale_laplacian(laplacian)
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

    def forward(self, x):
        batch_size, num_nodes, in_channels = x.size()
        L = self.laplacian.to(x.device)
        out = torch.zeros(batch_size, num_nodes, self.out_channels).to(x.device)
        Tx_0 = x
        Tx_1 = torch.bmm(L, x)
        out += torch.matmul(Tx_0, self.weights[0])
        if self.K > 1:
            out += torch.matmul(Tx_1, self.weights[1])
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(L, Tx_1) - Tx_0
            out += torch.matmul(Tx_2, self.weights[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out
