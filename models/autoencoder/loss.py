from torch import nn
from torch.nn import functional as F
import numpy as np


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self._history = []

    def forward(self, pred, target):
        n_b = pred.shape[0]

        loss = F.binary_cross_entropy_with_logits(pred, target)

        self._history.append((n_b, loss.cpu().item()))
        
        return loss


    def average(self):
        num_samples = np.sum([h[0] for h in self._history])

        return np.sum([h[0] * h[1] for h in self._history]) / num_samples
   
    def reset(self):
        self._history = []

    @property
    def history(self):
        return {'batch_size': [h[0] for h in self._history], 'value': [h[1] for h in self._history]}
