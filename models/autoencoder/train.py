import torch
import random
import numpy as np
from tqdm.auto import trange
import os

from .model import AutoEncoder
from .loss import BCE
from .metric import Accuracy
from .transform import no_transform, add_noise, binarize


def train(
        trainloader,
        valloader,
        config,
        ):
    exp_name, rootdir = config['name'], config['dir']
    model_params, checkpoint = config['model'], config['checkpoint']
    lr, num_epoch = config['lr'], config['epoch']

    model = AutoEncoder(**model_params)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])

    ##### WRITE YOUR CODE BELOW #####
    bce_loss = BCE()
    acc_metric = Accuracy()
    #################################

    model.train()
    pbar = trange(num_epoch, desc='Training autoencoder', leave=True)
    for epoch in pbar:
        for x in trainloader:
            ##### WRITE YOUR CODE BELOW #####
            x_ = binarize(x)
            y_target = no_transform(x)
            #################################

            with torch.set_grad_enabled(True):
                optim.zero_grad()
                y_pred = model(x_)

                ##### WRITE YOUR CODE BELOW #####
                loss = bce_loss(y_pred, y_target)
                #################################

                if torch.isnan(loss):
                    raise ValueError(f"NaN encountered in loss per mini-batch calculation, epoch {epoch}.")

                loss.backward()
                optim.step()

        model.eval()
        for x in valloader:
            ##### WRITE YOUR CODE BELOW #####
            x_ = binarize(x)
            y_target = no_transform(x)
            #################################

            with torch.no_grad():
                y_pred = model(x_)

                ##### WRITE YOUR CODE BELOW #####
                acc_metric(y_pred, y_target)
                #################################

        ##### WRITE YOUR CODE BELOW #####
        acc_mean = acc_metric.average()
        token = acc_metric.compare(acc_mean)

        pbar.set_description(f"Training autoencoder - Accuracy: {acc_mean}")

        acc_metric.reset()
        #################################
        
        if not os.path.exists(os.path.join(rootdir, exp_name)):
            os.mkdir(os.path.exists(os.path.join(rootdir, exp_name)))

        if token:
            torch.save({'model': model.state_dict()}, os.path.join(rootdir, exp_name, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({'model': model.state_dict()}, os.path.join(rootdir, exp_name, f"{epoch+1}.pt"))
