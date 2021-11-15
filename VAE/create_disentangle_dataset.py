"""
Created by Hamid Eghbal-zadeh at 22.03.21
Johannes Kepler University of Linz
"""

import argparse
import os
import pickle
import random

import numpy as np

from carla.env.env_rendering import EnvRenderer, get_gt_factors
from carla.train_agent import env_fnc
from contextual_gridworld.environment.colors import COLOR_TO_IDX, COLORS
from contextual_gridworld.environment.env import load_context_config
from general_utils.io_utils import check_dir

from PIL import Image
from tqdm import tqdm


def create_data(n_samples=50000, save_dir='', env=None, seed=None, tile_size=None, context_config=None, reward=None,
                grid_size=None, n_objects=None, root_path=None, alt_img_count=None):

    env_kwargs = dict(env_id=env, seed=seed, norm_obs=False, tile_size=tile_size,
                      context_config=context_config, reward=reward,
                      contextual=True, grid_size=grid_size, n_objects=n_objects)

    env = env_fnc(**env_kwargs)()
    max_n_obstacles = env.unwrapped.n_obstacles
    max_n_goodies = env.unwrapped.n_goodies
    total_objects = max_n_goodies + max_n_obstacles + 2  # +2 for agent and goal
    env_renderer = EnvRenderer(total_objects=total_objects, grid_size=grid_size,
                               tile_size=tile_size,
                               context_config=context_config)

    contexts, subdivs = load_context_config(context_config)
    valid_colors = list(set([COLOR_TO_IDX[v] for c in contexts for v in c.values()]))

    all_entities = [2, 3, 4, 5]

    OBJECT_DICT = {0: 'agent', 1: 'goal', 2: 'goodie'}
    sample_counter = {}
    for _ in tqdm(range(n_samples)):
        env.unwrapped.random_context = True
        env.unwrapped.random_object_positions = True

        env.unwrapped.n_obstacles = 0
        env.unwrapped.n_goodies = 1
        obs = env.reset()
        # env_info = env.unwrapped.get_gt_factors(fully_obs=args.fully_obs)
        env_info = get_gt_factors(env, total_objects, max_n_goodies, max_n_obstacles)
        gt, agent_pos, agent_dir = env_info
        valid_for_agent_pos, valid_for_gt = env_renderer.get_empty_positions(agent_pos, agent_dir)

        # set random agent pos and dir
        agent_pos = random.choice(valid_for_agent_pos)
        agent_dir = random.randint(0, 4)

        # compute new valid positions
        valid_for_agent_pos, valid_for_gt = env_renderer.get_empty_positions(agent_pos, agent_dir)

        gt_alt = gt.copy()
        # make sure all 3 entities exist
        for selected_entity in [0, 1, 2]:
            if all(gt[selected_entity::total_objects] > 0):
                selected_entity_exists = True
            else:
                selected_entity_exists = False
        if not selected_entity_exists:
            continue

        img = env_renderer.render_gt(gt, agent_pos, agent_dir)
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
                current_position = gt[selected_entity:: total_objects][0:2]
                current_valid_pos = []
                for p in valid_for_gt:
                    if any(current_position != p):
                        current_valid_pos.append(p)

                ix = np.random.choice(range(len(current_valid_pos)))
                selected_pos = current_valid_pos[ix]
                gt_alt[selected_entity:: total_objects][0:2] = selected_pos

                altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir)
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
                    gt_alt[selected_entity:: total_objects][0:2] = selected_pos
                    agent_pos_alt = selected_pos

                    altered_img = env_renderer.render_gt(gt_alt, agent_pos_alt, agent_dir)
                    altered_img_lst.append(altered_img)
                    altered_gt_lst.append(gt_alt)

            # change colour of agent
            else:
                altered_img_lst = []
                altered_gt_lst = []
                for alt_img_i in range(alt_img_count):
                    current_entity_color = gt[selected_entity:: total_objects][2]
                    current_valid_colors = list(set(valid_colors) - set([current_entity_color]))
                    selected_color = np.random.choice(current_valid_colors)
                    gt_alt[selected_entity:: total_objects][2] = selected_color

                    altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir)
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
                    current_position = gt[selected_entity:: total_objects][0:2]
                    current_valid_pos = []
                    for p in valid_for_gt:
                        if any(current_position != p):
                            current_valid_pos.append(p)

                    ix = np.random.choice(range(len(current_valid_pos)))
                    selected_pos = current_valid_pos[ix]
                    gt_alt[selected_entity:: total_objects][0:2] = selected_pos

                    altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir)
                    altered_img_lst.append(altered_img)
                    altered_gt_lst.append(gt_alt)
            # change colour of agent
            else:
                altered_img_lst = []
                altered_gt_lst = []
                for alt_img_i in range(alt_img_count):
                    current_entity_color = gt[selected_entity:: total_objects][2]
                    current_valid_colors = list(set(valid_colors) - set([current_entity_color]))
                    selected_color = np.random.choice(current_valid_colors)
                    gt_alt[selected_entity:: total_objects][2] = selected_color

                    altered_img = env_renderer.render_gt(gt_alt, agent_pos, agent_dir)
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
    parser.add_argument('--alt_img_count', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument("--context_config", help="which context configuration to load",
                        default='reasoning_contexts_train.yaml')
    parser.add_argument('--env', type=str, default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
    parser.add_argument('--grid_size', type=int, default=6)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument("--reward", help="choose reward configuration", default='pmlr.yaml')
    parser.add_argument('--root_path', type=str, default='')

    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--tile_size', type=int, default=12)
    args = parser.parse_args()

    create_data(args.n_samples, args.save_dir, args.env, args.seed, args.tile_size, args.context_config,
                args.reward, args.grid_size, args.n_objects, args.root_path, args.alt_img_count)
