import torch
from torch import nn
from torch.nn import functional as F


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

class KL(nn.Module):
    def __init__(self, beta=5e-4, gamma=1e-6):
        super(KL, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, mu, sigma):
        loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1/2).sum(dim=1).mean()
        loss = self.beta * loss
        self.beta *= (1 + self.gamma)
        
        return loss
