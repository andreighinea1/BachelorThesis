import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_probs=None):
        super(EmotionClassifier, self).__init__()

        if dropout_probs is None:
            dropout_probs = [0.5] * len(hidden_dims)
        if isinstance(dropout_probs, float):
            dropout_probs = [dropout_probs] * len(hidden_dims)

        # Add the input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dims[0]),  # TODO: Add BatchNorm1d back
            # nn.Dropout(p=dropout_probs[0]),  # TODO: Add back
        ]

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.extend([
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.ReLU(),
                # nn.BatchNorm1d(hidden_dims[i]),  # TODO: Add BatchNorm1d back
                # nn.Dropout(p=dropout_probs[i]),  # TODO: Add back
            ])

        # Add the output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
