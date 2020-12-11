from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Lambda(nn.Module):
    def __init__(self, _lambda):
        super(Lambda, self).__init__()
        self._lambda = _lambda

    def forward(self, x):
        return self._lambda(x)
