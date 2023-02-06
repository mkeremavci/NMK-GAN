import torch


def no_transform(x):
    return x


def add_noise(x):
    noise = torch.randn(x.shape) * .05
    x = x + noise

    return torch.clip(x, 0., 1.)
