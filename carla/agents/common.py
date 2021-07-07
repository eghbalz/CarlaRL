

import random
import time

import numpy as np

from carla.agents.core import ActorCritic
from carla.agents.utils.dummy_vec_env import DummyVecEnv
from carla.agents.utils.pytorch_utils import dict_obs_to_tensor, merge_dict_obs_list, merge_list_of_dicts
from carla.agents.utils.subproc_vec_env import SubprocVecEnv
from carla.agents.utils.weight_init import *


def logging(log_writer, logger, log_interval, epoch, steps_per_epoch, start_time):

    # log info to tensorboard
    log_writer.add_scalar(f'stats/EpRet', np.mean(logger.epoch_dict['EpRet']), epoch)
    log_writer.add_scalar(f'stats/EpLen', np.mean(logger.epoch_dict['EpLen']), epoch)
    log_writer.add_scalar(f'stats/LossPi', np.mean(logger.epoch_dict['LossPi']), epoch)
    log_writer.add_scalar(f'stats/LossV', np.mean(logger.epoch_dict['LossV']), epoch)
    log_writer.add_scalar(f'stats/Entropy', np.mean(logger.epoch_dict['Entropy']), epoch)

    # Log info about epoch
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)

    # log context statistics
    for key in logger.epoch_dict.keys():
        if "EpRet_" in key and len(logger.epoch_dict[key]) > 0:
            context = key.split("_")[-1]

            if epoch % log_interval == 0:
                log_writer.add_scalar(f'context_{context}/EpRet', np.mean(logger.epoch_dict[f"EpRet_{context}"]), epoch)
                log_writer.add_scalar(f'context_{context}/EpLen', np.mean(logger.epoch_dict[f"EpLen_{context}"]), epoch)
                log_writer.add_scalar(f'context_{context}/NObs', np.mean(logger.epoch_dict[f"NObs_{context}"]), epoch)
                log_writer.add_scalar(f'context_{context}/NGoodies', np.mean(logger.epoch_dict[f"NGoodies_{context}"]),
                                      epoch)
                log_writer.add_scalar(f'context_{context}/GoalReached',
                                      np.mean(logger.epoch_dict[f"GoalReached_{context}"]), epoch)

    logger.log_tabular('Time', time.time() - start_time)

    log_stats = logger.log_current_row
    print(f'Epoch: {epoch} | Avg. Ep. Return: {log_stats["AverageEpRet"]:.4f} '
          f'| Avg. Ep. Length: {log_stats["EpLen"]:.4f} | Time Passed: {log_stats["Time"]:.4f}')

    logger.dump_tabular(print_to_terminal=False)


def collect_epoch_data(ac, env, initial_obs, buf, local_steps_per_epoch, obs_space, device,
                       logger, n_proc, ep_ret, ep_len, max_ep_len, vae_buffer=None):

    # make sure agent is in eval mode, and for case of pretrained VAE, weights of encoder are frozen and in eval model.
    ac.eval()
    ac.freeze_context_net()
    ac.set_eval_context_net()

    o = initial_obs
    for t in range(local_steps_per_epoch):
        o = merge_dict_obs_list(o, obs_space)
        a, v, logp = ac.step(dict_obs_to_tensor(o, device=device))

        next_o, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1
        info = merge_list_of_dicts(info)

        # save and log
        buf.store(o, a, r, v, logp)
        logger.store(VVals=v)

        if vae_buffer is not None:
            vae_buffer.store(o)

        # Update obs (critical!)
        o = next_o
        for proc_idx in range(n_proc):
            timeout = ep_len[proc_idx] == max_ep_len
            terminal = d[proc_idx] or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:

                    if n_proc > 1:
                        # in case of more then one processes it should be wrapped as a list
                        step_o = [o[proc_idx]]
                    else:
                        step_o = o

                    _, v, _ = ac.step(dict_obs_to_tensor(merge_dict_obs_list(step_o, obs_space), device=device))
                    v = v[0]  # index 0 to get v as a single number
                else:
                    v = 0
                buf.finish_path(proc_idx, v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret[proc_idx], EpLen=ep_len[proc_idx])

                    # log context specific statistics
                    context_id = info['context'][proc_idx]
                    obstacles = info['obstacles'][proc_idx]
                    goodies = info['goodies'][proc_idx]
                    goal = info['goal_reached'][proc_idx]
                    logger.store(**{f'EpRet_{context_id}': ep_ret[proc_idx],
                                    f'EpLen_{context_id}': ep_len[proc_idx],
                                    f'NObs_{context_id}': obstacles,
                                    f'NGoodies_{context_id}': goodies,
                                    f'GoalReached_{context_id}': goal,
                                    })

                # no env reset necessary, handled implicitly by subroc_vec_env
                ep_ret[proc_idx] = 0
                ep_len[proc_idx] = 0

    # return the initial observation for the next epoch
    return o


def setup_agent(obs_space, action_space, ac_kwargs, device):

    # Create actor-critic module
    ac = ActorCritic(obs_space, action_space, **ac_kwargs)

    # handling freezing and eval mode for VAE in context_net
    ac.freeze_context_net()
    ac.set_eval_context_net()

    ac = ac.to(device)

    return ac


def setup_environments(env_fn, env_kwargs, eval_env_kwargs, n_proc):
    # test env for logging
    test_env = env_fn(rank=0, **env_kwargs)()

    # Instantiate environment
    env_fns = [env_fn(rank=i, **env_kwargs) for i in range(n_proc)]

    eval_env = env_fn(rank=0, **eval_env_kwargs)()

    if n_proc == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)

    return env, eval_env, test_env


def set_seeds(seed):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True