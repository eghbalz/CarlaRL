

import os
import pickle
import random

import numpy as np

from carla.env.env_rendering import EnvRenderer, get_gt_factors
from carla.train_agent import env_fnc
from PIL import Image
from tqdm import tqdm


def check_dir(directory):
    if not os.path.exists(directory):
        print('{} not exist. calling mkdir!'.format(directory))
        os.makedirs(directory)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MiniGrid-Contextual-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument('--context_config', help="which context configuration to load")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--tile_size', type=int, default=8)
    parser.add_argument('--reward', help="choose reward configuration")
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--n_objects', type=int, default=4)
    args = parser.parse_args()

    env_kwargs = dict(env_id=args.env, seed=args.seed, norm_obs=args.norm_obs, tile_size=args.tile_size,
                      context_config=args.context_config, reward=args.reward, contextual=True,
                      grid_size=args.grid_size, n_objects=args.n_objects)

    env = env_fnc(**env_kwargs)()

    max_n_goodies = env.unwrapped.n_goodies
    max_n_obstacles = env.unwrapped.n_obstacles
    total_objects = max_n_goodies + max_n_obstacles + 2  # +2 for agent and goal

    env_renderer = EnvRenderer(total_objects=total_objects, grid_size=args.grid_size,
                               tile_size=args.tile_size, context_config=args.context_config)

    sample_counter = {}
    for i in tqdm(range(args.n_samples)):

        env.unwrapped.random_context = True

        env.unwrapped.n_obstacles = np.random.randint(0, max_n_obstacles + 1)
        env.unwrapped.n_goodies = np.random.randint(0, max_n_goodies + 1)
        obs = env.reset()

        # set random agent direction
        env.unwrapped.agent_dir = np.random.randint(0, 4)
        gt, agent_pos, agent_dir = get_gt_factors(env.unwrapped, total_objects, max_n_goodies, max_n_obstacles)

        valid_pos, valid_pos_transformed = env_renderer.get_empty_positions(agent_pos, agent_dir)

        # set random agent position
        agent_pos = random.choice(valid_pos_transformed)

        img = env_renderer.render_gt(gt, agent_pos, agent_dir)

        context = np.argmax(obs['context'])
        context = str(context)
        if context in sample_counter:
            sample_counter[context] += 1
        else:
            sample_counter[context] = 0

        img_path = os.path.join(args.root_path, context)
        check_dir(img_path)

        im = Image.fromarray(img)
        im.save('{}/{}.jpg'.format(img_path, sample_counter[context]))
        with open('{}/{}.pkl'.format(img_path, sample_counter[context]), 'wb') as f:
            pickle.dump(gt, f)


