import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(3,2)

    def forward(self, state):
        return self.layer(state)

