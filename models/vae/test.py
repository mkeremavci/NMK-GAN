import torch
import numpy as np
from tqdm.auto import trange
import os


from .model import AutoEncoder


def test(
        testloader,
        config,
        device,
        ):
    exp_name, rootdir = config['name'], config['dir']
    model_params, checkpoint = config['model'], config['checkpoint']
    inference = config['inference']

    if checkpoint is None:
        raise ValueError('Checkpoint cannot be None for inference!')

    if inference != 'decode' and inference != 'encode':
        raise ValueError('Inference type must be one of the followings: encode, decode')

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict['model'])

    model.to(device)
    model.set_device(device)

    if not os.path.exists(os.path.join(rootdir, f"{exp_name}-test")):
        os.mkdir(os.path.join(rootdir, f"{exp_name}-test"))
    
    model.eval()
    preds = []
    for x in dataloader:
        with torch.no_grad()
            pred = model.encode(x) if inference == 'encode' else model.decode(x)
        preds += [p.numpy() for p in pred.cpu()]

    for i, p in enumerate(preds):
        np.savetxt(os.path.join(rootdir, f"{exp_name}-test", f"{inference}-{'0' * (6 - len(str(i)))}{i}.txt"))
