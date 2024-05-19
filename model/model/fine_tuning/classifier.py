import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.5):
        super(EmotionClassifier, self).__init__()

        # Add the input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        ]

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))

        # Add the output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
