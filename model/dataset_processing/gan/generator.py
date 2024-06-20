import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
        super(Generator, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.ConvTranspose1d(prev_dim, dim, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(True))
            prev_dim = dim
        layers.append(nn.ConvTranspose1d(prev_dim, output_dim, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        return self.main(x)
