import torch
from tqdm.auto import trange
import os

from .model import VAE
from .loss import BCE, KL
from .metric import Accuracy
from .transform import no_transform, add_noise, binarize


def train(
        trainloader,
        valloader,
        config,
        device,
        ):
    exp_name, rootdir = config['name'], config['dir']
    model_params, checkpoint = config['model'], config['checkpoint']
    lr, num_epoch = config['lr'], config['epoch']

    model = VAE(**model_params)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint is not None:
        print(f"Checkpoint {checkpoint.split('/')[-1:]} is loading...")
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'])

    model.to(device)
    model.set_device(device)

    ##### WRITE YOUR CODE BELOW #####
    bce_loss = BCE()
    kl_div = KL(beta=5e-4, gamma=1e-6)
    acc_metric = Accuracy()
    #################################

    model.train()
    pbar = trange(num_epoch, desc='Training VAE', leave=True)
    for epoch in pbar:
        for x in trainloader:
            ##### WRITE YOUR CODE BELOW #####
            x_ = no_transform(x)
            y_target = binarize(x)

            x_, y_target = x_.to(device), y_target.to(device)
            #################################

            with torch.set_grad_enabled(True):
                optim.zero_grad()
                mu, sigma = model.encoder(x_)
                z = model.sample(mu, sigma)
                y_pred = model.decoder(z)

                ##### WRITE YOUR CODE BELOW #####
                rc_loss = bce_loss(y_pred, y_target)
                kl_loss = kl_div(mu, sigma)
                loss = rc_loss + kl_loss
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

            x_, y_target = x_.to(device), y_target.to(device)
            #################################

            with torch.no_grad():
                y_pred = model(x_)

                ##### WRITE YOUR CODE BELOW #####
                acc_metric(y_pred, y_target)
                #################################

        ##### WRITE YOUR CODE BELOW #####
        rc_mean = bce_loss.average()
        kl_mean = kl_div.average()
        acc_mean = acc_metric.average()
        token = acc_metric.compare(acc_mean)

        pbar.set_description(f"Training VAE - Accuracy: {acc_mean} - RecLoss: {rc_mean} - KLdiv: {kl_mean}")

        bce_loss.reset()
        kl_div.reset()
        acc_metric.reset()
        #################################
        
        if not os.path.exists(os.path.join(rootdir, exp_name)):
            os.mkdir(os.path.join(rootdir, exp_name))

        if token:
            torch.save({'model': model.state_dict()}, os.path.join(rootdir, exp_name, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({'model': model.state_dict()}, os.path.join(rootdir, exp_name, f"{epoch+1}.pt"))
