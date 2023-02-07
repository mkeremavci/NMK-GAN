import torch
import numpy as np


class Accuracy:
    def __init__(self):
        self._history = []
        self._best = 0

    def __call__(self, pred, target):
        n_b = pred.shape[0]

        pred = (torch.sigmoid(pred) > 0.5).to(torch.uint8)
        pred = pred[:].contiguous().view(-1)
        target = target[:].contiguous().view(-1)

        acc = ((pred == target).sum() / len(pred)).cpu()
        self._history.append((n_b, acc))

        return acc

    def average(self):
        num_samples = np.sum([h[0] for h in self._history])

        return np.sum([h[0] * h[1] for h in self._history]) / num_samples
   
    def reset(self):
        self._history = []

    def compare(self, acc):
        if acc > self._best:
            self._best = acc
            return True
        return False

    @property
    def history(self):
        return {'batch_size': [h[0] for h in self._history], 'value': [h[1] for h in self._history]}

    @property
    def best(self):
        return self._best
