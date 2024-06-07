import torch
from torch import nn
from torch.nn import init


class BatchDenseChebConv(nn.Module):
    r"""Chebyshev Spectral Graph Convolution layer from `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__

    We recommend to use this module when applying ChebConv on dense graphs.

    Parameters
    ----------
    in_feats: int
        Dimension of input features :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k: int
        Chebyshev filter size.
    activation: str, optional
        Activation function. Default: ``"relu"``.
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import torch
    >>> from model.fine_tuning.batch_dense_cheb_conv import BatchDenseChebConv
    >>>
    >>> feat = torch.ones(6, 10)
    >>> adj = torch.tensor([[0., 0., 1., 0., 0., 0.],
    ...                     [1., 0., 0., 0., 0., 0.],
    ...                     [0., 1., 0., 0., 0., 0.],
    ...                     [0., 0., 1., 0., 0., 1.],
    ...                     [0., 0., 0., 1., 0., 0.],
    ...                     [0., 0., 0., 0., 0., 0.]])
    >>> conv = BatchDenseChebConv(10, 2, 2)
    >>> res = conv(adj, feat)
    >>> res
    tensor([[-0.8028, -1.7049],
            [-0.8028, -1.7049],
            [-0.8028, -1.7049],
            [-2.6551, -1.1607],
            [ 0.5070, -2.0897],
            [ 3.6690, -3.0187]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_feats, out_feats, k, activation="relu", bias=True):
        super(BatchDenseChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k

        self.W = nn.Parameter(torch.Tensor(k, in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)

        if activation:
            self._activation = activation
        else:
            self._activation = "relu"

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias)
        for i in range(self._k):
            init.xavier_normal_(self.W[i], init.calculate_gain(self._activation))

    def forward(self, feat, adj, lambda_max=None):
        r"""Compute (Dense) Chebyshev Spectral Graph Convolution layer

        Parameters
        ----------
        feat: torch.Tensor
            The input feature of shape :math:`(B, N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        adj: torch.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(B, N, N)`, where B is the batch size,
            N is the number of nodes, a row represents the destination
            and a column represents the source.
        lambda_max: float or None, optional
            A float value indicates the largest eigenvalue of given graph.
            Default: None.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(B, N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        A = adj.to(feat)
        B, N, _ = A.shape

        # Normalizing adjacency matrix for each graph in the batch
        in_degree = 1 / A.sum(dim=-1).clamp(min=1).sqrt()  # Shape: (B, N)
        D_invsqrt = torch.diag_embed(in_degree)  # Shape: (B, N, N)
        I = torch.eye(N).to(A).unsqueeze(0).expand(B, -1, -1)  # Shape: (B, N, N)
        L = I - D_invsqrt @ A @ D_invsqrt  # Shape: (B, N, N)

        # Scaling Laplacian
        if lambda_max is None:
            lambda_: torch.Tensor = torch.linalg.eigvals(L).real  # Shape: (B, N)
            lambda_max = lambda_.max(dim=-1)[0]  # Shape: (B)
            # lambda_max[:, None, None] -> Shape: (B, 1, 1)

        L_hat = 2 * L / lambda_max[:, None, None] - I  # Shape: (B, N, N)

        # Compute Chebyshev polynomials
        Z = [I]
        if self._k > 1:
            Z.append(L_hat)
        for i in range(2, self._k):
            Z.append(2 * L_hat @ Z[-1] - Z[-2])

        Zs = torch.stack(Z, 0)  # Shape: (k, B, N, N)

        # Apply Chebyshev polynomials to the input feature
        # (k, B, N, N) @ (B, N, D_in) => (k, B, N, D_in)
        Zh = torch.einsum('kbnm,bmd->kbnd', Zs, feat)  # Shape: (k, B, N, D_in)
        # (k, B, N, D_in) @ (k, D_in, D_out) => (k, B, N, D_out)
        Zh = torch.einsum('kbnd,kdo->kbno', Zh, self.W)  # Shape: (k, B, N, D_out)
        Zh = Zh.sum(0)  # Shape: (B, N, D_out)

        if self.bias is not None:
            Zh = Zh + self.bias  # Shape: (B, N, D_out)
        return Zh
