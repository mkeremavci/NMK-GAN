import argparse
import torch
import torchvision
import random
import numpy as np
import importlib
import json
import os

SEED = 16
AVAILABLE = {
        'mnist': torchvision.datasets.MNIST,
        'flowers': torchvision.datasets.Flowers102,
        }

##### WRITE YOUR CODE BELOW #####
def collate_fn(batch):
    collated = [torchvision.transforms.functional.to_tensor(x[0].resize((32, 32))) for x in batch]
    
    return torch.stack(collated)
#################################


def set_seeds(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Train a model',
        description='Train a model with the given arguments',
        )
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-b', '--batchsize', required=True)

    args = parser.parse_args()

    set_seeds(SEED)

    if args.dataset.lower() not in list(AVAILABLE.keys()):
        raise ValueError('Given dataset not available!')
    dataset = AVAILABLE[args.dataset.lower()]

    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')

    try:
        trainset = dataset(root='data/', train=True, download=False)
        valset = dataset(root='data/', train=False, download=False)
    except:
        trainset = dataset(root='data/', split='train', download=False)
        valset = dataset(root='data/', split='val', download=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batchsize), shuffle=True, collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(valset, batch_size=int(args.batchsize), shuffle=False, collate_fn=collate_fn)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train = importlib.import_module(f"models.{args.model}.train").train

    train(trainloader, valloader, config, device)
