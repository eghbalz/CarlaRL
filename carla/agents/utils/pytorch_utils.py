import torch

import numpy as np


def dict_obs_to_tensor(obs, device):

    tensor_obs = {}
    for space in obs.keys():
        tensor_obs[space] = torch.as_tensor(obs[space], dtype=torch.float32, device=device).contiguous()

    return tensor_obs


def merge_dict_obs_list(obs, obs_space):

    if not isinstance(obs, dict):
        obs_dict = {}
        for space in obs_space:
            obs_dict[space] = np.stack([o[space] for o in obs])
        obs = obs_dict

    return obs


def merge_list_of_dicts(list_of_dicts):

    if not isinstance(list_of_dicts, dict):
        merged_dict = {}
        for key in list_of_dicts[0]:
            merged_dict[key] = np.stack([d[key] for d in list_of_dicts])
    else:
        merged_dict = list_of_dicts

    return merged_dict