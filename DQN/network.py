import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3,128),
            nn.ReLU(),
            nn.Linear(128,2),
        )

    def forward(self, state):
        return self.network(state)