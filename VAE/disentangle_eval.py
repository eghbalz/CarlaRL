"""
Created by Hamid Eghbal-zadeh at 22.03.21
Johannes Kepler University of Linz
"""

import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import argparse
import pickle
import matplotlib.pyplot as plt
import json

from datasets.utils import get_disentangled_loaders
from carla.architectures.utils import get_model
from general_utils.io_utils import check_dir

from sklearn import linear_model
from sklearn.metrics import classification_report

import pickle
from carla.env.wrapper import *

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


def _prune_dims(variances, threshold=0.):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def eval(img_size=64, batch_size=128, lr=0.005, weight_decay=0.0, scheduler_gamma=0.95, num_epochs=100,
         ds_name='celeba', n_workers=8, crop_size=148,
         latent_dim=256, save_dir='', milestones=[25, 75], schedule_type='exp', aug=True,
         dec_out_nonlin='tanh', init='he', vae_model_path='', use_posterior=False, alt_img_count=1,
         env=None, seed=None, fully_obs=None, tile_size=None, context_config=None, reward=None, grid_size=None,
         n_objects=None, vae_uid=None,
         args_dict=None

         ):
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

    vae_model_path = os.path.join(vae_model_path, vae_uid)
    encoder = load_and_set_vae_weights(vae_model_path)
    encoder.to(device)

    trn_loader, num_train_imgs = get_disentangled_loaders(ds_name, crop_size, img_size, batch_size,
                                                          n_workers, aug, dec_out_nonlin, alt_img_count)



    log_dict = {'trn': {'loss': [], 'LR': []},
                'val': {'loss': [], 'LR': []}}
    random_state = np.random.RandomState(0)
    model = linear_model.LogisticRegression(random_state=random_state)
    features, labels, embeddings, gts = [], [], [], []

    for data in tqdm(trn_loader):
        X1, X2s, gt1, gt2, y = data
        X1, y = X1.to(device), y.to(device)
        with torch.no_grad():
            emb1_mu, emb1_var = encoder.encode(X1)
            if use_posterior:
                emb1 = encoder.reparameterize(emb1_mu, emb1_var)
            else:
                emb1 = emb1_mu.detach().cpu().numpy()

        feat_mean = []
        for X2 in X2s:
            X2 = X2.to(device)
            if (X1 - X2).abs().sum() < 0.00001:
                print('found near-duplicates!')
                for b_i in range(X1.shape[0]):
                    plt.subplot(1, 2, 1)
                    plt.imshow(X1[b_i].cpu().permute(1, 2, 0))
                    plt.subplot(1, 2, 2)
                    plt.imshow(X2[b_i].cpu().permute(1, 2, 0))
                    plt.title('batch index {} class {}'.format(b_i, y))
                    plt.pause(2)
                    plt.close(0)
                continue
            with torch.no_grad():
                emb2_mu, emb2_var = encoder.encode(X2)
                if use_posterior:
                    emb2 = encoder.reparameterize(emb2_mu, emb2_var)
                else:
                    emb2 = emb2_mu.detach().cpu().numpy()

                # feat = np.mean(np.abs(emb1 - emb2), axis=0)
                feat = np.abs(emb1 - emb2)
                feat_mean.append(feat)
                # if sum(feat) <= 0.0001:
                #     continue
        features.append(np.mean(feat_mean, 0))
        labels.append(y.detach().cpu().numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    n_class = len(np.unique(labels))

    num_ds = features.shape[0]
    valid_size = 0.2
    indices = list(range(num_ds))
    num_val_imgs = int(np.floor(valid_size * num_ds))
    num_train_imgs = num_ds - num_val_imgs
    np.random.shuffle(indices)
    train_idx, val_idx = indices[num_val_imgs:], indices[:num_val_imgs]

    train_features, train_labels = features[train_idx], labels[train_idx]
    val_features, val_labels = features[val_idx], labels[val_idx]
    model.fit(train_features, train_labels)

    train_preds = model.predict(train_features)
    print(classification_report(train_labels, train_preds))

    val_preds = model.predict(val_features)
    print(classification_report(val_labels, val_preds))



print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-latent-dim', type=int, default=100)
    parser.add_argument('-dec-out-nonlin', choices=['tanh', 'sig'], default="sig")
    parser.add_argument('-init', choices=['kaiming', 'xavier', 'none'], default="none")

    parser.add_argument('-lr', type=float, default=.0003)
    parser.add_argument('-weight-decay', type=float, default=0)  # 5e-4
    parser.add_argument('-schedule-type', choices=['cos', 'exp', 'step', 'none'], default="none")
    parser.add_argument('-scheduler-gamma', type=float, default=0.5)
    parser.add_argument('-milestones', nargs='+', default=[50, 75], type=int)

    parser.add_argument('-img-size', type=int, default=84)

    parser.add_argument('-crop-size', type=int, default=84)
    parser.add_argument('-num-epochs', type=int, default=50)
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-ds-name', type=str,
                        default="carla4fully48pxdisentangle")
    parser.add_argument('-alt_img_count', type=int, default=1)
    parser.add_argument('-n-workers', type=int, default=0)

    parser.add_argument('-aug', nargs='+', default=[''], type=str)
    parser.add_argument('-save-dir', type=str, required=True)
    parser.add_argument('-vae-model-path', type=str, default='')
    parser.add_argument('-use-posterior', default=False, action='store_true')

    parser.add_argument('--env', type=str, default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fully_obs', default=False, action='store_true')
    parser.add_argument('--random_start', default=False, action='store_true')
    parser.add_argument('--no_goodies', default=False, action='store_true')
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument("--context_config", help="which context configuration to load", default='color_contexts.yaml')
    parser.add_argument('--tile_size', type=int, default=12)
    parser.add_argument('--random_goal', default=False, action='store_true')
    parser.add_argument("--reward", help="choose reward configuration", default='default.yaml')
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('-vae-uid', type=str, default='')


    args = parser.parse_args()

    eval(args.img_size, args.batch_size, args.lr, args.weight_decay, args.scheduler_gamma, args.num_epochs,
         args.ds_name, args.n_workers, args.crop_size, args.latent_dim, args.save_dir, args.milestones,
         args.schedule_type, args.aug, args.dec_out_nonlin, args.init, args.vae_model_path, args.use_posterior,
         args.alt_img_count,
         args.env, args.seed, args.fully_obs, args.tile_size, args.context_config, args.reward, args.grid_size,
         args.n_objects, args.vae_uid,
         args_dict=args.__dict__)
