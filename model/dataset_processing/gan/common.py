import torch.nn as nn


# noinspection PyMethodMayBeStatic
class LastStep(nn.Module):
    def __init__(self):
        super(LastStep, self).__init__()

    def forward(self, x):
        return x[:, -1, :]
