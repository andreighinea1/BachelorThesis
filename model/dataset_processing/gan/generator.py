import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
        super(Generator, self).__init__()
        layers = []
        prev_dim = input_dim

        # Intermediate layers
        for dim in layer_dims:
            layers.append(nn.ConvTranspose1d(prev_dim, dim, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(True))
            prev_dim = dim

        layers.append(nn.ConvTranspose1d(prev_dim, output_dim, kernel_size=4, stride=4, padding=0, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        print("Generator - Input size:", x.size())
        for layer in self.main:
            x = layer(x)
            if "ConvTranspose1d" in layer.__class__.__name__:
                print(layer.__class__.__name__, "output size:", x.size())
        return x
