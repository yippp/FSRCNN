import torch.nn as nn
from torch import mean, sqrt, pow

class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.SmoothL1Loss = nn.SmoothL1Loss()
        return

    def forward(self, input, target):
        loss = self.SmoothL1Loss(input / self.delta, target / self.delta)
        return loss * self.delta * self.delta

class CharbonnierLoss(nn.Module):
    def __init__(self, delta=1e-3):
        super(CharbonnierLoss, self).__init__()
        # self.MSELoss = nn.MSELoss ()
        self.delta = delta
        return

    def forward(self, input, target):
        # return torch.sqrt(self.MSELoss(input, target) + self.delta * self.delta)
        return mean(sqrt(pow((input - target), 2) + self.delta * self.delta))