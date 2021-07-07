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
from random import shuffle
from datasets.utils import get_disentangled_loaders
from carla.architectures.utils import get_model, get_clf_model
from general_utils.io_utils import check_dir

from torch import nn
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from carla.train_agent import env_fnc
import pickle
from carla.env.env_rendering import EnvRenderer
from carla.env.wrapper import *

from contextual_gridworld.environment.colors import COLOR_TO_IDX, COLORS
from contextual_gridworld.environment.env import load_context_config
from PIL import Image
import pickle


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
         dec_out_nonlin='tanh', init='he', vae_model_path='', use_posterior=False, seq_len=50,
         env=None, seed=None, fully_obs=None, tile_size=None, context_config=None, reward=None, grid_size=None,
         n_objects=None, root_path=None, alt_img_count=None,
         args_dict=None

         ):
    env_kwargs = dict(env_id=env, seed=seed, fully_obs=fully_obs, no_goodies=False,
                      random_start=True, norm_obs=False, tile_size=tile_size,
                      context_config=context_config, random_goal=False, reward=reward,
                      contextual=True, grid_size=grid_size, n_objects=n_objects)

    env = env_fnc(**env_kwargs)()

    env_renderer = EnvRenderer(total_objects=env.unwrapped.total_objects, grid_size=grid_size,
                               tile_size=tile_size, agent_view_size=env.unwrapped.agent_view_size,
                               context_config=context_config)
    max_n_obstacles = env.unwrapped.max_n_obstacles
    max_n_goodies = env.unwrapped.max_n_goodies
    max_color = COLORS.__len__() - 1
    max_pos = env.unwrapped.agent_view_size - 1
    contexts, subdivs = load_context_config(context_config)
    # valid_positions=
    valid_colors = list(set([COLOR_TO_IDX[v] for c in contexts for v in c.values()]))

    plt.ion()
    # all_entities = [1, 2, 3, 4, 5]
    # all_changable_entities = [2, 3, 4, 5]
    all_entities = [2, 3, 4, 5]

    OBJECT_DICT = {0: 'agent', 1: 'goal', 2: 'goodie'}
    sample_counter = {}
    for _ in tqdm(range(390)):
        for _ in range(batch_size):
            env.unwrapped.random_context = True
            env.unwrapped.random_object_positions = True

            env.unwrapped.n_obstacles = 0
            env.unwrapped.n_goodies = 1
            obs = env.reset()
            env_info = env.unwrapped.get_gt_factors(fully_obs=args.fully_obs)
            gt, agent_pos, agent_dir = env_info
            valid_for_agent_pos, valid_for_gt = env_renderer.get_empty_positions(agent_pos, agent_dir,
                                                                                 fully_obs=args.fully_obs)
            gt_alt = gt.copy()
            # make sure all 3 entities exist
            for selected_entity in [0, 1, 2]:
                if all(gt[selected_entity:: env.unwrapped.total_objects] > 0):
                    selected_entity_exists = True
                else:
                    selected_entity_exists = False
            if not selected_entity_exists:
                continue

            img = env_renderer.render_gt(gt, agent_pos, agent_dir, fully_obs=args.fully_obs)
            context_id = np.argwhere(obs['context'] == 1).squeeze().astype('int')

            # randomly decide which object (agent:0, goal:1, goodie:2)
            selected_entity = np.random.choice([0, 1, 2])
            # decide what variable to change (location/colour)

            # for goal, only change location (only corners)
            if selected_entity == 1:
                factor = 'location'
                entity_name = OBJECT_DICT[selected_entity]
                altered_img_lst = []
                altered_gt_lst = []
                for alt_img_i in range(alt_img_count):
                    current_position = gt[selected_entity:: env.unwrapped.total_objects][0:2]
                    current_valid_pos = []
                    for p in valid_for_gt:
                        if any(current_position != p):
                            current_valid_pos.append(p)

                    ix = np.random.choice(range(len(current_valid_pos)))
                    selected_pos = current_valid_pos[ix]
                    gt_alt[selected_entity:: env.unwrapped.total_objects][0:2] = selected_pos

                    altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir, fully_obs=args.fully_obs)
                    altered_img_lst.append(altered_img)
                    altered_gt_lst.append(gt_alt)

            # for agent, decide colour or location
            elif selected_entity == 0:
                entity_name = OBJECT_DICT[selected_entity]
                factor = np.random.choice(['colour', 'location'])
                # change location of agent
                if factor == 'location':
                    altered_img_lst = []
                    altered_gt_lst = []
                    for alt_img_i in range(alt_img_count):
                        current_valid_pos = []
                        for p in valid_for_agent_pos:
                            if any(agent_pos != p):
                                current_valid_pos.append(p)

                        ix = np.random.choice(range(len(current_valid_pos)))
                        selected_pos = current_valid_pos[ix]
                        gt_alt[selected_entity:: env.unwrapped.total_objects][0:2] = selected_pos
                        agent_pos_alt = selected_pos

                        altered_img = env_renderer.render_gt(gt_alt, agent_pos_alt, agent_dir, fully_obs=args.fully_obs)
                        altered_img_lst.append(altered_img)
                        altered_gt_lst.append(gt_alt)

                # change colour of agent
                else:
                    altered_img_lst = []
                    altered_gt_lst = []
                    for alt_img_i in range(alt_img_count):
                        current_entity_color = gt[selected_entity:: env.unwrapped.total_objects][2]
                        current_valid_colors = list(set(valid_colors) - set([current_entity_color]))
                        selected_color = np.random.choice(current_valid_colors)
                        gt_alt[selected_entity:: env.unwrapped.total_objects][2] = selected_color

                        altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir, fully_obs=args.fully_obs)
                        altered_img_lst.append(altered_img)
                        altered_gt_lst.append(gt_alt)

            # for goodie, decide colour or location
            elif selected_entity == 2:
                entity_name = OBJECT_DICT[selected_entity]
                factor = np.random.choice(['colour', 'location'])
                # print('{} {}'.format(entity_name, factor))

                # change location of agent
                if factor == 'location':
                    altered_img_lst = []
                    altered_gt_lst = []
                    for alt_img_i in range(alt_img_count):
                        current_position = gt[selected_entity:: env.unwrapped.total_objects][0:2]
                        current_valid_pos = []
                        for p in valid_for_gt:
                            if any(current_position != p):
                                current_valid_pos.append(p)

                        ix = np.random.choice(range(len(current_valid_pos)))
                        selected_pos = current_valid_pos[ix]
                        gt_alt[selected_entity:: env.unwrapped.total_objects][0:2] = selected_pos

                        altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir, fully_obs=args.fully_obs)
                        altered_img_lst.append(altered_img)
                        altered_gt_lst.append(gt_alt)
                # change colour of agent
                else:
                    altered_img_lst = []
                    altered_gt_lst = []
                    for alt_img_i in range(alt_img_count):
                        current_entity_color = gt[selected_entity:: env.unwrapped.total_objects][2]
                        current_valid_colors = list(set(valid_colors) - set([current_entity_color]))
                        selected_color = np.random.choice(current_valid_colors)
                        gt_alt[selected_entity:: env.unwrapped.total_objects][2] = selected_color

                        altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir, fully_obs=args.fully_obs)
                        altered_img_lst.append(altered_img)
                        altered_gt_lst.append(gt_alt)

            assert altered_img_lst.__len__() == alt_img_count

            label = str('{}_{}'.format(entity_name, factor))
            if label in sample_counter:
                sample_counter[label] += 1
            else:
                sample_counter[label] = 0

            img_path = os.path.join(root_path, label)
            check_dir(img_path)


            im = Image.fromarray(img)
            im.save('{}/{}.jpg'.format(img_path, sample_counter[label]))
            with open('{}/{}.pkl'.format(img_path, sample_counter[label]), 'wb') as f:
                pickle.dump(gt, f)

            for i, altered_img in enumerate(altered_img_lst):
                im = Image.fromarray(altered_img)
                im.save('{}/{}.jpg_alt{}'.format(img_path, sample_counter[label], i + 1), format='JPEG')
                with open('{}/{}.pkl_alt{}'.format(img_path, sample_counter[label], i + 1), 'wb') as f:
                    pickle.dump(altered_gt_lst[i], f)

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
    parser.add_argument('-ds-name',
                        choices=['carla4fully48pxdisentangle'],
                        default="carla4fully48pxdisentangle")
    parser.add_argument('-seq-len', type=int, default=50)
    parser.add_argument('-n-workers', type=int, default=8)

    parser.add_argument('-aug', nargs='+', default=[''], type=str)
    parser.add_argument('-save-dir', type=str, required=True)
    parser.add_argument('-vae-model-path', type=str, default='')
    parser.add_argument('-use-posterior', default=False, action='store_true')

    parser.add_argument('--env', type=str, default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--fully_obs', default=False, action='store_true')
    parser.add_argument('--random_start', default=False, action='store_true')
    parser.add_argument('--no_goodies', default=False, action='store_true')
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument("--context_config", help="which context configuration to load",
                        default='reasoning_contexts_train.yaml')
    parser.add_argument('--tile_size', type=int, default=12)
    parser.add_argument('--random_goal', default=False, action='store_true')
    parser.add_argument("--reward", help="choose reward configuration", default='pmlr.yaml')
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--alt_img_count', type=int, default=3)


    args = parser.parse_args()

    eval(args.img_size, args.batch_size, args.lr, args.weight_decay, args.scheduler_gamma, args.num_epochs,
         args.ds_name, args.n_workers, args.crop_size, args.latent_dim, args.save_dir, args.milestones,
         args.schedule_type, args.aug, args.dec_out_nonlin, args.init, args.vae_model_path, args.use_posterior,
         args.seq_len,
         args.env, args.seed, args.fully_obs, args.tile_size, args.context_config, args.reward, args.grid_size,
         args.n_objects, args.root_path, args.alt_img_count,
         args_dict=args.__dict__)
