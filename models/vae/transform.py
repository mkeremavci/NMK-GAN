import torch


def no_transform(x):
    return x


def add_noise(x, sigma=.05):
    noise = torch.randn(x.shape) * sigma
    x = x + noise

    return torch.clip(x, 0., 1.)


def binarize(x, thresh=0.5):
    return (x > thresh).to(torch.float32)
