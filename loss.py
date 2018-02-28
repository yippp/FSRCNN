import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.SmoothL1Loss = nn.SmoothL1Loss()
        return

    def forward(self, input, target):
        loss = self.SmoothL1Loss(input / self.delta, target / self.delta)
        return loss * self.delta * self.delta