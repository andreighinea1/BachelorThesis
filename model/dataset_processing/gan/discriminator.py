import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0.3):
        super(Discriminator, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Conv1d(prev_dim, dim, 4, 2, 1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Conv1d(prev_dim, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
