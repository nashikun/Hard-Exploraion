import torch
from torch import nn


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.exp(x)))
