from torch import nn
from torch.nn import functional as F


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)
