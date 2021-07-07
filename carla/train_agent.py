import copy
import gym
import os
import warnings

import torch.nn as nn

from gym.wrappers.filter_observation import FilterObservation
from carla.agents.utils.logx import setup_logger_kwargs
from carla.agents.vpg import vpg
from carla.env.wrapper import *

# import to register environments
import contextual_gridworld.environment


def env_fnc(seed, env_id, rank=0, contextual=False, max_ep_length=100, norm_obs=False, reward='default.yaml',
            tile_size=8, context_config='color_contexts.yaml', grid_size=8, n_objects=4):

    def build_env():
        # we can directly pass some parameters to the environment here
        env = gym.make(env_id, max_steps=max_ep_length, seed=seed + rank, reward_config=reward,
                       context_config=context_config, grid_size=grid_size, n_objects=n_objects)

        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        filter_keys = ['image']

        if contextual:
            filter_keys.append('context')

        env = FilterObservation(env, filter_keys)

        env = RGBImgObsWrapper(env, tile_size=tile_size)
        env = RGBImgObsRotationWrapper(env)

        if norm_obs:
            normalization = {'image': 255}
            env = NormalizeObservationByKey(env, normalization)

        return env

    return build_env


def setup_arguments():

    DEFAULT_CONTEXT = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'env', 'context_configurations', 'pmlr_all.yaml')
    DEFAULT_REWARD = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'env', 'reward_configurations', 'pmlr.yaml')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MiniGrid-Contextual-v0')
    parser.add_argument('--hid_policy_net', nargs='+', help='list of neurons in each layer in policy net',
                        default=[64, 64])
    parser.add_argument('--hid_context_net', nargs='+', help='list of neurons in each layer in context net.',
                        default=[512])
    parser.add_argument('--hid_bottleneck', type=int, default=128)
    parser.add_argument('--hid_state_net', nargs='+', help='list of neurons in each layer in state net.',
                        default=[16, 16, 32, 32])
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--conditioning', default='none')
    parser.add_argument('--gamma', type=float, default=0.97)

    # vpg specific parameters
    parser.add_argument('--pi_lr', type=float, default=7e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_proc', help='number of processes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=3000)


    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--entropy_coef', type=float, default=0.01)

    parser.add_argument('--contextual', default=False, action='store_true')
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument('--max_ep_length', type=int, default=100)
    parser.add_argument('--reward', help='choose reward configuration', default=DEFAULT_REWARD)
    parser.add_argument('--context_config', help="which context configuration to load", default=DEFAULT_CONTEXT)
    parser.add_argument('--tile_size', type=int, default=8)
    parser.add_argument('--grid_size', type=int, default=6)
    parser.add_argument('--n_objects', type=int, default=4)

    # VAE
    parser.add_argument('--use_posterior', default=False, action='store_true')
    parser.add_argument('--context_encoder_model_path', type=str, default='')
    parser.add_argument('--state_encoder_model_path', type=str, default='')
    parser.add_argument('--normalize_context_vector', default=False, action='store_true')
    parser.add_argument('--vae_bottleneck', type=int, default=100)
    parser.add_argument('--vae_train', default=False, action='store_true')
    parser.add_argument('--state_encoder_freeze', default=False, action='store_true')

    # logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo experiment')

    # evaluation
    parser.add_argument('--eval_context_config', help="which context configuration to load for evaluation",
                        default=DEFAULT_CONTEXT)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--eval_episodes', type=int, default=20)

    # vae parameters
    parser.add_argument('--vae_kernel-sizes', nargs='+', default=[6, 4, 3, 3], type=int)
    parser.add_argument('--vae_output-padding-lst', nargs='+', default=[0, 0, 0, 0], type=int)
    parser.add_argument('--vae_strides', nargs='+', default=[2, 1, 1, 1], type=int)
    parser.add_argument('--vae_paddings', nargs='+', default=[0, 0, 0, 0], type=int)
    parser.add_argument('--vae_hidden-dims', nargs='+', default=[32, 32, 64, 64],
                        type=int)
    parser.add_argument('--vae_latent-dim', type=int, default=100)
    parser.add_argument('--vae_enc-bn', default=False, action='store_true')
    parser.add_argument('--vae_dec-bn', default=False, action='store_true')

    parser.add_argument('--vae_dec-out-nonlin', choices=['tanh', 'sig', 'none'], default="none")
    parser.add_argument('--vae-reduction', choices=['mean', 'norm_batch', 'norm_pixel', 'norm_dim', 'norm_rel', 'sum'],
                        default="norm_batch")
    parser.add_argument('--vae_nonlin', choices=['relu', 'lrelu', 'elu'], default="lrelu")
    parser.add_argument('--vae_prior', choices=['gauss', 'bernoulli'], default="gauss")
    parser.add_argument('--vae_soft-clip', default=False, action='store_true')
    parser.add_argument('--vae_loss-type', choices=['beta', 'annealed', 'iw', 'geco'], default="beta")

    parser.add_argument('--vae_img-size', type=int, default=48)
    parser.add_argument('--vae_batch-size', type=int, default=128)
    parser.add_argument('--vae_init', choices=['kaiming', 'xavier', 'none'], default="kaiming")
    parser.add_argument('--vae-lr', type=float, default=5e-4)

    # have to be set
    parser.add_argument('--vae-gamma', type=float, default=10.)
    parser.add_argument('--vae-c-max', type=float, default=0.005)
    parser.add_argument('--vae-c-stop-iter', type=iter, default=100)
    parser.add_argument('--vae-beta', type=float, default=1.)
    parser.add_argument('--vae-geco-goal', type=float, default=0.5)

    parser.add_argument('--joint_vae', default=False, action='store_true')
    parser.add_argument('--vae_update_interval', type=int, default=10)
    parser.add_argument('--vae_epochs', type=int, default=30)
    parser.add_argument('--vae_buffer_size', type=int, default=50000)


    args = parser.parse_args()

    vae_params = {
        "kernel_sizes": args.vae_kernel_sizes,
        "output_padding_lst": args.vae_output_padding_lst,
        "strides": args.vae_strides,
        "paddings": args.vae_paddings,
        "hidden_dims": args.vae_hidden_dims,
        "latent_dim": args.vae_latent_dim,
        "enc_bn": args.vae_enc_bn,
        "dec_bn": args.vae_dec_bn,
        "dec_out_nonlin": args.vae_dec_out_nonlin,
        "vae_reduction": args.vae_reduction,
        "nonlin": args.vae_nonlin,
        "prior": args.vae_prior,
        "soft_clip": args.vae_soft_clip,
        "loss_type": args.vae_loss_type,
        "vae_gamma": args.vae_gamma,
        "vae_c_max": args.vae_c_max,
        "vae_beta": args.vae_beta,
        "vae_geco_goal": args.vae_geco_goal,
        "img_size": args.vae_img_size,
        "batch_size": args.vae_batch_size,
        "init": args.vae_init,
    }

    # handling empty lists
    if args.hid_policy_net == ['0']:
        args.hid_policy_net = []
    if args.hid_context_net == ['0']:
        args.hid_context_net = []
    if args.hid_state_net == ['0']:
        args.hid_state_net = []

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if not args.contextual:
        if args.hid_context_net != ['0']:
            warnings.warn('hid_context_net is not [0]! corrected.')
            args.hid_context_net = []

        if args.conditioning != 'none':
            warnings.warn('conditioning is not none! corrected.')
            args.context = 'none'
    else:
        if args.conditioning == 'none' and args.hid_context_net != []:
            warnings.warn('conditioning none needs hid_context_net=[]! corrected.')
            args.hid_context_net = []

        if args.conditioning != 'none':
            if args.hid_context_net == []:
                warnings.warn('conditioning {} needs hid_context_net==[hid_bottleneck]! corrected.')
                args.hid_context_net = [args.hid_bottleneck]

            if args.hid_context_net[-1] != args.hid_bottleneck:
                warnings.warn('conditioning {} needs hid_context_net[-1]==hid_bottleneck! corrected.')

    env_kwargs = dict(env_id=args.env, seed=args.seed, contextual=args.contextual, max_ep_length=args.max_ep_length,
                      norm_obs=args.norm_obs, reward=args.reward, tile_size=args.tile_size,
                      context_config=args.context_config, grid_size=args.grid_size, n_objects=args.n_objects)

    eval_env_kwargs = copy.deepcopy(env_kwargs)
    eval_env_kwargs['context_config'] = args.eval_context_config

    ac_kwargs = dict(hidden_sizes_policy_net=[int(i) for i in args.hid_policy_net],
                     hidden_sizes_context_net=[int(i) for i in args.hid_context_net],
                     hidden_sizes_state_net=[int(i) for i in args.hid_state_net],
                     hidden_bottleneck=args.hid_bottleneck,
                     conditioning=args.conditioning,
                     contextual=args.contextual,
                     activation=nn.ReLU,
                     kernel_size=args.kernel_size,
                     stride=args.stride,
                     use_posterior=args.use_posterior,
                     context_encoder_model_path=args.context_encoder_model_path,
                     state_encoder_model_path=args.state_encoder_model_path,
                     normalize_context_vector=args.normalize_context_vector,
                     vae_bottleneck=args.vae_bottleneck,
                     vae_train=args.vae_train,
                     state_encoder_freeze=args.state_encoder_freeze,
                     joint_training=args.joint_vae,
                     vae_params=vae_params)

    return args, env_kwargs, eval_env_kwargs, ac_kwargs, logger_kwargs


if __name__ == '__main__':

    args, env_kwargs, eval_env_kwargs, ac_kwargs, logger_kwargs = setup_arguments()

    vpg(env_fnc=env_fnc, env_kwargs=env_kwargs, eval_env_kwargs=eval_env_kwargs,
        ac_kwargs=ac_kwargs, seed=args.seed, logger_kwargs=logger_kwargs, log_interval=args.log_interval,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_v_iters=args.train_v_iters, epochs=args.epochs,
        steps_per_epoch=args.steps, max_ep_len=args.max_ep_length, n_proc=args.n_proc,
        gamma=args.gamma, entropy_coef=args.entropy_coef, vae_buffer_size=args.vae_buffer_size,
        vae_lr=args.vae_lr, vae_epochs=args.vae_epochs, vae_update_interval=args.vae_update_interval)

