import torch
from tqdm.auto import trange

from .model import AutoEncoder
from .loss import BCE
from .transform import no_transform, add_noise


def train(
        dataloader,
        config,
        ):
    model_params, checkpoint, lr, num_epoch = config['model'], config['checkpoint'], config['lr'], config['epoch']

    model = AutoEncoder(**model_params)
    optim = torch.optim.Adam(lr=lr)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])

    ##### WRITE YOUR CODE BELOW #####
    bce_loss = BCE()

    model.train()
    
    for epoch in trange(num_epoch, desc='Training autoencoder'):
        for x in dataloader:
            with torch.set_grad_enabled(True):
                pred = model(transform(x))
                loss = torch.Tensor

            

        pass

    
    pass