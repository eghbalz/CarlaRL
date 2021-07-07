"""
adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py
"""

import glob
import os
import tqdm

import carla.agents.core as core

from carla.agents.common import *
from carla.agents.buffer import MultipleWorkerBuffer, VAEBuffer, VAEDataset
from carla.agents.eval_utils import evaluate_agent
from carla.agents.utils.logx import EpochLogger

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def update(ac, optim_pi, optim_v, train_v_iters, entropy_coef, buffer, logger):
    """ RL update code """

    # set agent train mode
    ac.train()
    ac.freeze_context_net()
    ac.set_eval_context_net()

    # data
    data = buffer.get()
    obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'],  data['ret']

    optim_pi.zero_grad()

    # Policy loss
    pi, logp, _ = ac(obs, act)
    loss_pi = -(logp * adv).mean()

    ent = pi.entropy().mean()

    (loss_pi - entropy_coef*ent).backward()

    optim_pi.step()

    losses_v = []

    for i in range(train_v_iters):
        optim_v.zero_grad()
        _, _, v = ac(obs)
        loss_v = ((v - ret) ** 2).mean()
        loss_v.backward()
        optim_v.step()

        losses_v.append(loss_v.item())

    logger.store(LossPi=loss_pi.item(), LossV=np.mean(losses_v), Entropy=ent.item())


def update_vae(ac, optimizer, n_epochs, buffer, log_writer, epoch, device):
    """ VAE update code """

    print(f'Training VAE for {n_epochs} epoch(s)...')
    data = buffer.get()

    model = ac.state_context_net.context_encoder
    batch_size = model.batch_size

    dataset = VAEDataset(data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    num_train_imgs = data.shape[0]

    log_dict = {'loss': [], 'Reconstruction_Loss': [], 'KLD': []}

    ac.train()
    ac.state_context_net.unfreeze_vae()

    for ep in range(n_epochs):
        acc_trn_loss = {'loss': [], 'Reconstruction_Loss': [], 'KLD': []}

        for X in tqdm.tqdm(dataloader):

            results = model(X.to(device))
            train_loss, train_loss_dict = model.loss_function(*results, M_N=batch_size / num_train_imgs)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            for k in acc_trn_loss.keys():
                acc_trn_loss[k].append(train_loss_dict[k].item())

        for k in log_dict:
            if k != 'LR':
                log_dict[k].append(np.mean(acc_trn_loss[k]))

    for key in log_dict:
        log_writer.add_scalar(f'vae_training/{key}', np.mean(log_dict[key]), epoch)

    print('Done training VAE...')


def vpg(env_fnc, env_kwargs, eval_env_kwargs, ac_kwargs=dict(), seed=0, logger_kwargs=dict(), save_freq=10,
        device='cuda', log_interval=10, eval_interval=100, eval_episodes=5,
        pi_lr=7e-4, vf_lr=1e-3, train_v_iters=30, epochs=100, steps_per_epoch=4096, max_ep_len=100, n_proc=1,
        gamma=0.97, entropy_coef=0.0,  vae_buffer_size=50000, vae_lr=5e-4, vae_epochs=1, vae_update_interval=1):
    """
    (Almost) Vanilla Policy Gradient
    adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py

    :param env_fnc: environment creation function
    :param env_kwargs: environment parameters
    :param eval_env_kwargs: evaluation environment parameters (e.g. with different context configuration)
    :param ac_kwargs: actor critic parameters
    :param seed: random seed for reproducibility
    :param logger_kwargs: additional logger arguments
    :param save_freq: how often to store model parameters
    :param device: device to train on (GPU or CPU)
    :param log_interval: how often to log statistics
    :param eval_interval: how often the agent should be evaluated, e.g. every 50 epochs
    :param eval_episodes: how many episodes to evaluate the agent
    :param pi_lr: policy learning rate
    :param vf_lr: value function learning rate
    :param train_v_iters: number of iterations the value function will be updated per policy update
    :param epochs: number of epochs to train
    :param steps_per_epoch: how many environment steps are taken per epoch
    :param max_ep_len: maximum length of an epoch
    :param n_proc: number of parallel processes
    :param gamma: discounting factor
    :param entropy_coef: entropy regularization coefficient
    :param vae_buffer_size: size of the VAE buffer (for Online-Carla)
    :param vae_lr: VAE learning rate (for Online-Carla)
    :param vae_epochs: number of epochs the VAE will be updated (for Online-Carla)
    :param vae_update_interval: how often VAE will be updated, e.g. every 15 epochs (for Online-Carla)
    """

    device = torch.device(device)

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    log_writer = SummaryWriter(log_dir=logger_kwargs['output_dir'])

    # remove old log
    for old_log in glob.glob(os.path.join(logger_kwargs['output_dir'], "events.*")):
        os.unlink(old_log)

    # set random seeds
    set_seeds(seed)

    steps_per_epoch = steps_per_epoch
    n_proc = n_proc
    epochs = epochs

    env, eval_env, test_env = setup_environments(env_fnc, env_kwargs, eval_env_kwargs, n_proc)
    act_dim = env.action_space.shape
    obs_space = env.observation_space

    # Create actor-critic module
    ac = setup_agent(obs_space, env.action_space, ac_kwargs, device)

    # model summary
    logger.log('\nmodel summary: \n%s\n' % ac)

    # Count variables
    var_counts = core.count_vars(ac)
    logger.log('\nNumber of parameters:  \t %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / n_proc)
    buf = MultipleWorkerBuffer(obs_space, act_dim, local_steps_per_epoch, device,
                               gamma=gamma, n_proc=n_proc)

    # Set up optimizers for policy and value function
    optim_pi = Adam(list(ac.state_context_net.parameters()) + list(ac.policy_logits_net.parameters()),
                    lr=pi_lr, eps=1e-5)
    optim_v = Adam(list(ac.state_context_net.parameters()) + list(ac.policy_v_net.parameters()), lr=vf_lr, eps=1e-5)

    obs_buffer = None
    if ac.joint_training:
        obs_buffer = VAEBuffer(vae_buffer_size)
        ac.state_context_net.unfreeze_vae()
        optim_vae = Adam(ac.state_context_net.context_encoder.parameters(), lr=vae_lr, weight_decay=0.0)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), np.zeros(n_proc), np.zeros(n_proc)

    eval_step = 0
    max_score = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o = collect_epoch_data(ac, env, o, buf, local_steps_per_epoch, obs_space, device, logger,
                               n_proc, ep_ret, ep_len, max_ep_len, vae_buffer=obs_buffer)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': test_env}, None)

        # Perform VPG update!
        update(ac, optim_pi, optim_v, train_v_iters, entropy_coef, buf, logger)

        if ac.joint_training:
            if (epoch + 1) % vae_update_interval == 0:
                update_vae(ac, optim_vae, vae_epochs, obs_buffer, log_writer, epoch, device=device)

        logging(log_writer, logger, log_interval, epoch, steps_per_epoch, start_time)

        # evaluate agent
        if (epoch + 1) % eval_interval == 0:
            score = evaluate_agent(eval_env, ac, eval_episodes, device, log_writer, eval_step)

            if score >= max_score:
                max_score = score
                logger.save_state({'env': eval_env}, "best")

            logger.save_state({'env': eval_env}, eval_step)
            eval_step += 1

        ac.set_gamma(min(epoch/(epochs//2), 1))

    # shutdown environment
    env.close()
