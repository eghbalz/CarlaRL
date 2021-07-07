
import argparse
import os

import gym

from gym_minigrid.window import Window
from carla.env.wrapper import *

import contextual_gridworld.environment


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=32)

    window.show_img(img)


def reset():

    obs = env.reset()

    print(f'Context: {env.unwrapped.context}')
    window.set_caption(f'Context: {env.unwrapped.context}')

    redraw(obs['image'])


def step(action):
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))
    if done:
        print('done!')
        reset()
    else:
        redraw(obs['image'])


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default='MiniGrid-Contextual-v0')
parser.add_argument('--agent_view', default=False, help="draw the agent sees (partially observable view)",
                    action='store_true')
parser.add_argument("--reward", help="choose reward configuration",
                    default=os.path.join('env', 'reward_configurations', 'pmlr.yaml'))
parser.add_argument("--context_config", help="which context configuration to load",
                    default=os.path.join('env', 'context_configurations', 'pmlr_all.yaml'))
parser.add_argument("--grid_size", type=int, help="size of the grid world", default=6)
parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
parser.add_argument('--n_objects', type=int, default=4)
parser.add_argument('--context_id', type=int, default=-1)

args = parser.parse_args()

env = gym.make(args.env, reward_config=args.reward, grid_size=args.grid_size,
               context_config=args.context_config, n_objects=args.n_objects)

if args.context_id != -1:
    env.random_context = False
    env.context_id = args.context_id

if args.agent_view:

    env = RGBImgObsWrapper(env, tile_size=12)
    env = RGBImgObsRotationWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
