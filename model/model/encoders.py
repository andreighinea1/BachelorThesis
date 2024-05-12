import torch.nn as nn


class TimeFrequencyEncoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=64, num_layers=2, nhead=8):
        super(TimeFrequencyEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # To match the model dimension if necessary
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.fc(x)  # Reduce dimension to model_dim
        return self.transformer_encoder(x)


class CrossSpaceProjector(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(CrossSpaceProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projector(x)
