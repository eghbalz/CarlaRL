"""
Created by Hamid Eghbal-zadeh at 04.02.21
Johannes Kepler University of Linz
"""
"""
Created by Hamid Eghbal-zadeh at 20.12.20
Johannes Kepler University of Linz
"""
import torch
from torch import optim

import numpy as np
import os
from datetime import datetime
import argparse
import pickle
import matplotlib.pyplot as plt
import json

from datasets.utils import get_dataset_loaders
from carla.architectures.utils import get_model, get_clf_model
from general_utils.io_utils import check_dir

from torch import nn


msg_template = 'ep {} loss: ({:.5f}, {:.5f}) acc: ({:.2f}, {:.2f}) lr: {}'


def load_and_set_vae_weights(vae_model_path):
    model_path = os.path.join(vae_model_path, 'model.pt')
    args_path = os.path.join(vae_model_path, 'args.pkl')

    with open(args_path, 'rb') as handle:
        args_dict = pickle.load(handle)

    context_encoder = get_model(**args_dict)
    context_encoder.load_state_dict(torch.load(model_path))
    context_encoder = freez_vae(context_encoder)
    context_encoder = set_eval_vae(context_encoder)
    return context_encoder


def freez_vae(context_encoder):
    for param in context_encoder.parameters():
        param.requires_grad = False
    return context_encoder


def set_eval_vae(context_encoder):
    context_encoder.eval()
    return context_encoder


def forward(model, encoder, X, clf_type, use_posterior):
    if clf_type == 'linear' or clf_type == 'mlp':
        with torch.no_grad():
            emb_mu, emb_var = encoder.encode(X)
            if use_posterior:
                emb = encoder.reparameterize(emb_mu, emb_var)
            else:
                emb = emb_mu
        y_hat = model(emb)
    elif 'cnn' in clf_type or 'vgg' in clf_type or 'resnet' in clf_type:
        y_hat = model(X)
    return y_hat


def eval(img_size=64, batch_size=128, lr=0.005, weight_decay=0.0, scheduler_gamma=0.95, num_epochs=100,
         ds_name='celeba', n_workers=8, crop_size=148,
         latent_dim=256, save_dir='', milestones=[25, 75], schedule_type='exp', aug=True,
         dec_out_nonlin='tanh', init='he', vae_model_path='', vae_uid=None, use_posterior=False, clf_type='linear',
         opt='adam',n_context=90,
         args_dict=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)

    path = os.path.join(save_dir, ds_name, uid)
    check_dir(path)
    with open(os.path.join(path, 'args.pkl'), 'wb') as handle:
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path, 'args.txt'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    device = torch.device('cuda')
    if clf_type in ['linear', 'mlp']:
        encoder = load_and_set_vae_weights(os.path.join(vae_model_path, ds_name, vae_uid))
        encoder.to(device)
    else:
        encoder = None
    model = get_clf_model(init, latent_dim, clf_type, n_context=n_context)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=weight_decay)
    if schedule_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif schedule_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma)
    elif schedule_type == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    trn_loader, val_loader, num_train_imgs, num_val_imgs = get_dataset_loaders(ds_name, crop_size, img_size, batch_size,
                                                                               n_workers, aug, dec_out_nonlin)



    log_dict = {'trn': {'loss': [], 'LR': []},
                'val': {'loss': [], 'LR': []}}

    print('start training ...')
    for ep in range(num_epochs):
        acc_trn_loss = []
        acc_val_loss = []
        model.train()
        correct_trn, total_trn = 0, 0
        for data in trn_loader:
            X, y = data
            X, y = X.to(device), y.to(device)
            y_hat = forward(model, encoder, X, clf_type, use_posterior)
            train_loss = criterion(y_hat, y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            acc_trn_loss.append(train_loss.item())

            _, predicted = torch.max(y_hat, 1)
            correct_trn += (predicted == y).sum().item()
            total_trn += y.size(0)

        if schedule_type != 'none':
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        model.eval()
        correct_val, total_val = 0, 0
        for data in val_loader:
            X, y = data
            X, y = X.to(device), y.to(device)
            y_hat = forward(model, encoder, X, clf_type, use_posterior)
            val_loss = criterion(y_hat, y)
            acc_val_loss.append(val_loss.item())

            _, predicted = torch.max(y_hat, 1)
            correct_val += (predicted == y).sum().item()
            total_val += y.size(0)

        msg = msg_template.format(ep, np.mean(acc_trn_loss), np.mean(acc_val_loss),
                                  correct_trn / total_trn * 100, correct_val / total_val * 100, lr_now)
        print(msg)

    fname = 'clf_model.pt'

    print('saving model ...')
    torch.save(model.state_dict(), os.path.join(path, fname))

    print('saving measure plots ...')
    for k in log_dict['trn']:
        plt.plot(log_dict['trn'][k], label='trn ' + k)
        plt.plot(log_dict['val'][k], label='val ' + k)
        plt.legend()
        plt.savefig(os.path.join(path, k + '.jpg'))
        plt.clf()

    print('saving logs ...')
    with open(os.path.join(path, 'logs.pkl'), 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-latent-dim', type=int, default=100)
    parser.add_argument('-dec-out-nonlin', choices=['tanh', 'sig'], default="none")
    parser.add_argument('-init', choices=['kaiming', 'xavier', 'none'], default="none")

    parser.add_argument('-lr', type=float, default=.0005)
    parser.add_argument('-weight-decay', type=float, default=0)  # 5e-4
    parser.add_argument('-schedule-type', choices=['cos', 'exp', 'step', 'none'], default="none")
    parser.add_argument('-scheduler-gamma', type=float, default=0.5)
    parser.add_argument('-milestones', nargs='+', default=[50, 75], type=int)

    parser.add_argument('-img-size', type=int, default=84)

    parser.add_argument('-crop-size', type=int, default=148)
    parser.add_argument('-num-epochs', type=int, default=50)
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-ds-name', type=str, default="carla45fully48px")
    parser.add_argument('-n-context', type=int, default=90)
    parser.add_argument('-n-workers', type=int, default=8)

    parser.add_argument('-aug', nargs='+', default=[''], type=str)
    parser.add_argument('-save-dir', type=str, required=True)
    parser.add_argument('-vae-model-path', type=str, default='')
    parser.add_argument('-vae-uid', type=str, required=True)
    parser.add_argument('-use-posterior', default=False, action='store_true')

    parser.add_argument('-clf-type',
                        choices=['linear', 'mlp'],
                        default="linear")
    parser.add_argument('-opt',
                        choices=['adam', 'sgd'],
                        default="adam")

    args = parser.parse_args()

    eval(args.img_size, args.batch_size, args.lr, args.weight_decay, args.scheduler_gamma, args.num_epochs,
         args.ds_name, args.n_workers, args.crop_size, args.latent_dim, args.save_dir, args.milestones,
         args.schedule_type, args.aug, args.dec_out_nonlin, args.init, args.vae_model_path, args.vae_uid,
         args.use_posterior, args.clf_type, args.opt, args.n_context,
         args_dict=args.__dict__)
