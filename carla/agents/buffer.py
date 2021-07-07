"""
adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py
"""

import torch

import carla.agents.core as core
import numpy as np

from collections import deque
from PIL import Image
from torch.utils.data import Dataset


class MultipleWorkerBuffer:
    """
    A buffer for storing trajectories experienced of a multiple worker agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_space, act_dim, size, device, gamma=0.99, lam=1., n_proc=1):
        self.device = device
        self.obs_space = obs_space

        self.obs_dim = {}
        self.obs_buf = {}

        for space in self.obs_space:
            obs_dim = self.obs_space[space].shape
            self.obs_dim[space] = obs_dim
            self.obs_buf[space] = np.zeros((n_proc,) + core.combined_shape(size, obs_dim), dtype=np.float32)

        self.act_buf = np.zeros((n_proc,) + core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((n_proc, size), dtype=np.float32)
        self.rew_buf = np.zeros((n_proc, size), dtype=np.float32)
        self.ret_buf = np.zeros((n_proc, size), dtype=np.float32)
        self.val_buf = np.zeros((n_proc, size), dtype=np.float32)
        self.logp_buf = np.zeros((n_proc, size), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = np.zeros(n_proc, dtype=np.int32), np.zeros(n_proc,
                                                                                                  dtype=np.int32), size
        self.n_proc = n_proc

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert (self.ptr < self.max_size).all()  # buffer has to have room so you can store

        for i, ptr in enumerate(self.ptr):

            for space in obs.keys():
                self.obs_buf[space][i][ptr] = obs[space][i]

            self.act_buf[i][ptr] = act[i]
            self.rew_buf[i][ptr] = rew[i]
            self.val_buf[i][ptr] = val[i]
            self.logp_buf[i][ptr] = logp[i]
            self.ptr[i] += 1

    def finish_path(self, proc_idx, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx[proc_idx], self.ptr[proc_idx])

        rews = np.append(self.rew_buf[proc_idx][path_slice], last_val)
        vals = np.append(self.val_buf[proc_idx][path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[proc_idx][path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[proc_idx][path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx[proc_idx] = self.ptr[proc_idx]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert (self.ptr == self.max_size).all()  # buffer has to be full before you can get

        self.ptr, self.path_start_idx = np.zeros(self.n_proc, dtype=np.int32), np.zeros(self.n_proc, dtype=np.int32)

        obs = {}
        # flatten over all processes
        for space in self.obs_buf.keys():
            obs[space] = torch.as_tensor(self.obs_buf[space].reshape(self.n_proc * self.max_size, *self.obs_dim[space]), dtype=torch.float32, device=self.device)

        act = self.act_buf.reshape(self.n_proc * self.max_size)
        ret = self.ret_buf.reshape(self.n_proc * self.max_size)
        adv = self.adv_buf.reshape(self.n_proc * self.max_size)
        logp = self.logp_buf.reshape(self.n_proc * self.max_size)

        # the next two lines implement the advantage normalization trick (add small epsilon to avoid division with 0)
        adv_mean, adv_std = np.mean(adv), np.std(adv)
        adv = (adv - adv_mean) / (adv_std + 1e-5)
        data = dict(act=act, ret=ret, adv=adv, logp=logp)
        data = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}
        data['obs'] = obs
        return data


class VAEBuffer:
    """
    Buffer for storing observations to train Online-Carla
    """
    def __init__(self, size):

        self.obs_buf = deque(maxlen=size)

    def store(self, obs):

        imgs = obs['image']

        for img in imgs:
            self.obs_buf.append(img)

    def get(self):

        obs = np.stack(self.obs_buf)
        return obs


class VAEDataset(Dataset):
    """
    Dataset to train Online-Carla
    """

    def __init__(self, data, transform=None):

        self.data = data
        self.length = data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        sample = self.data[item]
        if self.transform is not None:
            # some transforms require PIL, but since the data is already
            # normalized it has to be un-normalized before transformation (weird!)
            # normalization is done by the transform
            sample = Image.fromarray(np.uint8(sample * 255))
            sample = self.transform(sample)

        return sample