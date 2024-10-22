import torch.nn as nn

from dataset_processing.gan.common import LastStep


class Generator(nn.Module):
    def __init__(self, latent_dim, conv_dims, lstm_dim, eeg_length, use_full_lstm=True):
        super(Generator, self).__init__()
        self.eeg_length = eeg_length
        self.latent_dim = latent_dim

        # Upsample layer
        layers = [nn.Upsample(scale_factor=30)]

        prev_dim = 1
        for i, (conv_dim, kernel_size, stride, padding) in enumerate(conv_dims):
            layers.append(nn.Conv1d(prev_dim, conv_dim, kernel_size, stride, padding, bias=False))
            layers.append(nn.BatchNorm1d(conv_dim, eps=0.001, momentum=0.99))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = conv_dim

            if i == 0:
                layers.append(nn.MaxPool1d(8, stride=8, padding=4))
                layers.append(nn.Dropout(p=0.5))

        layers.append(nn.MaxPool1d(4, stride=4, padding=2))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.LSTM(16, lstm_dim, batch_first=True))

        self.block1 = nn.Sequential(*layers)

        self.block2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(lstm_dim * lstm_dim if use_full_lstm else lstm_dim, eeg_length),
            nn.Tanh(),
        )
        if not use_full_lstm:
            self.block2.insert(0, LastStep())

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, self.latent_dim))  # Reshape to (batch_size, 1, latent_dim)

        x = self.block1(x)[0]
        x = self.block2(x)

        x = x.view(x.size(0), 1, self.eeg_length)  # Reshape to (batch_size, 1, eeg_length)
        return x
