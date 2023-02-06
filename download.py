import argparse
import os
import torchvision


AVAILABLE = {
        'mnist': torchvision.datasets.MNIST,
        'flowers': torchvision.datasets.Flowers102,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='download',
        description='download datasets for training',
        )
    parser.add_argument('-d', '--dataset', required=True)

    args = parser.parse_args()

    if args.dataset.lower() not in list(AVAILABLE.keys()):
        raise ValueError()

    if not os.path.exists('data'):
        os.mkdir('data')

    dataset = AVAILABLE[args.dataset.lower()]

    dataset(
        root='data/',
        download=True,
        )
