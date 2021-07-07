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
from general_utils.io_utils import check_dir
from datasets.utils import get_dataset_loaders_vae
from carla.architectures.utils import get_model


msg_template = 'ep {} loss: {:.10f} \trecon: {:.10f} \tkld loss: {:.10f} \tkld: {:.10f} \tlr: {}'


def train(img_size=64, batch_size=128, lr=0.005, weight_decay=0.0, scheduler_gamma=0.95, num_epochs=100,
          ds_name='celeba', n_workers=8, crop_size=148, loss_type='H',
          vae_gamma=10, vae_beta=1, kernel_sizes=[4, 4, 4, 4],
          output_padding_lst=[],
          latent_dim=256, hidden_dims=[32, 64, 128, 256, 512], nonlin='relu', enc_bn=False,
          dec_bn=False, save_dir='', milestones=[25, 75], schedule_type='exp', aug=True,
          dec_out_nonlin='tanh', prior='gauss', soft_clip=False, init='he', strides=[], paddings=[], vae_c_max=25,
          vae_c_stop_iter=100,
          vae_geco_goal=0.5, vae_reduction='mean',
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

    model = get_model(kernel_sizes, output_padding_lst,
                      latent_dim, hidden_dims, img_size, loss_type, vae_gamma, vae_beta, nonlin, enc_bn,
                      dec_bn,
                      dec_out_nonlin, prior, soft_clip, init, strides, paddings, batch_size, vae_c_max, vae_c_stop_iter,
                      vae_geco_goal,
                      vae_reduction)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if schedule_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif schedule_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma)

    trn_loader, num_train_imgs = get_dataset_loaders_vae(ds_name, crop_size, img_size, batch_size,
                                                         n_workers, aug, dec_out_nonlin)



    log_dict = {'trn': {'loss': [], 'Reconstruction_Loss': [], 'KLD': [], 'KLD_Raw': [], 'LR': []},
                }

    print('start training ...')
    for ep in range(num_epochs):
        acc_trn_loss = {'loss': [], 'Reconstruction_Loss': [], 'KLD': [], 'KLD_Raw': []}
        model.train()

        for data in trn_loader:
            X, y = data
            results = model(X.to(device))
            train_loss, train_loss_dict = model.loss_function(*results, M_N=batch_size / num_train_imgs)


            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            for k in acc_trn_loss.keys():
                if k == 'KLD_Raw':
                    with torch.no_grad():
                        kld = model.kl_div(mu=results[2], log_var=results[3]).mean()
                    acc_trn_loss[k].append(kld.item())
                else:
                    acc_trn_loss[k].append(train_loss_dict[k].item())

        if loss_type == 'annealed':
            model.num_iter += 1
        if schedule_type != 'none':
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']

        for k in log_dict['trn']:
            if k != 'LR':
                log_dict['trn'][k].append(np.mean(acc_trn_loss[k]))
        log_dict['trn']['LR'].append(lr_now)

        msg = msg_template.format(ep, np.mean(acc_trn_loss['loss']), np.mean(
            acc_trn_loss['Reconstruction_Loss']), np.mean(acc_trn_loss['KLD']), np.mean(acc_trn_loss['KLD_Raw']),
                                  lr_now)
        print(msg)

    fname = 'model.pt'

    print('saving model ...')
    torch.save(model.state_dict(), os.path.join(path, fname))

    print('saving measure plots ...')
    for k in log_dict['trn']:
        plt.plot(log_dict['trn'][k], label='trn ' + k)

        plt.legend()
        plt.savefig(os.path.join(path, k + '.jpg'))
        plt.clf()

    print('saving logs ...')
    with open(os.path.join(path, 'logs.pkl'), 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-kernel-sizes', nargs='+', default=[6, 4, 3, 3], type=int)
    parser.add_argument('-output-padding-lst', nargs='+', default=[0, 0, 0, 0], type=int)
    parser.add_argument('-strides', nargs='+', default=[2, 1, 1, 1], type=int)
    parser.add_argument('-paddings', nargs='+', default=[0, 0, 0, 0], type=int)
    parser.add_argument('-hidden-dims', nargs='+', default=[32, 32, 64, 64],
                        type=int)
    parser.add_argument('-latent-dim', type=int, default=256)
    parser.add_argument('--enc-bn', default=False, action='store_true')
    parser.add_argument('--dec-bn', default=False, action='store_true')

    parser.add_argument('-dec-out-nonlin', choices=['tanh', 'sig', 'none'], default="tanh")
    parser.add_argument('-vae-reduction', choices=['mean', 'norm_batch', 'norm_pixel', 'norm_dim', 'norm_rel', 'sum'],
                        default="mean")
    parser.add_argument('-prior', choices=['gauss', 'bernoulli'], default="gauss")
    parser.add_argument('--soft-clip', default=False, action='store_true')
    parser.add_argument('-loss-type', choices=['beta', 'annealed', 'geco'], default="beta")
    parser.add_argument('-vae-gamma', type=float, default=10.)
    parser.add_argument('-vae-c-max', type=float, default=0.005)
    parser.add_argument('-vae-c-stop-iter', type=int, default=80)  # 50000*100/128
    parser.add_argument('-vae-beta', type=float, default=1.)
    parser.add_argument('-vae-geco-goal', type=float, default=6)
    parser.add_argument('-init', choices=['kaiming', 'xavier', 'none'], default="none")

    parser.add_argument('-lr', type=float, default=.0003)
    parser.add_argument('-weight-decay', type=float, default=0)
    parser.add_argument('-schedule-type', choices=['exp', 'step', 'none'], default="none")
    parser.add_argument('-scheduler-gamma', type=float, default=0.5)
    parser.add_argument('-milestones', nargs='+', default=[50, 75], type=int)

    parser.add_argument('-img-size', type=int, default=48)

    parser.add_argument('-crop-size', type=int, default=64)
    parser.add_argument('-num-epochs', type=int, default=50)
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-ds-name', choices=['carla45fully48px'], default="carla45fully48px")

    parser.add_argument('-n-workers', type=int, default=8)
    parser.add_argument('-nonlin', choices=['relu', 'lrelu', 'elu'], default="lrelu")

    parser.add_argument('-aug', nargs='+', default=[''], type=str)
    parser.add_argument('-save-dir', type=str, required=True)

    args = parser.parse_args()

    train(args.img_size, args.batch_size, args.lr, args.weight_decay, args.scheduler_gamma, args.num_epochs,
          args.ds_name, args.n_workers, args.crop_size, args.loss_type,
          args.vae_gamma, args.vae_beta, args.kernel_sizes, args.output_padding_lst,
          args.latent_dim, args.hidden_dims, args.nonlin, args.enc_bn, args.dec_bn, args.save_dir, args.milestones,
          args.schedule_type, args.aug, args.dec_out_nonlin, args.prior, args.soft_clip, args.init, args.strides,
          args.paddings, args.vae_c_max, args.vae_c_stop_iter, args.vae_geco_goal, args.vae_reduction,
          args_dict=args.__dict__)
