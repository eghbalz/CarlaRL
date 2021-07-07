import torch
import warnings

import numpy as np

from carla.agents.utils.logx import EpochLogger
from carla.agents.utils.pytorch_utils import dict_obs_to_tensor, merge_dict_obs_list


def evaluation_metric(goal, normalized_remaining_obs, normalized_remaining_goodies, w_G=0.5, w_OG=0.5):

    # between 0 and 1
    G = goal
    RG = normalized_remaining_goodies
    RO = normalized_remaining_obs

    # between -1 and 1
    object_eval = RO - RG

    # between 0 and 1
    object_eval = np.maximum(object_eval, 0)

    return (w_G * G) + (w_OG * object_eval)


def evaluate_agent(env, ac, num_episodes, device, log_writer, eval_step):
    ac.eval()
    ac.freeze_context_net()
    ac.set_eval_context_net()

    observation_space = env.observation_space

    n_contexts = env.unwrapped.n_contexts

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = dict_obs_to_tensor(merge_dict_obs_list([x], observation_space), device)
            action = ac.act(x)
        return action

    logger = EpochLogger()

    env.unwrapped.random_context = False

    # evaluate agent in each context for num_episodes
    for context_id in range(n_contexts):
        env.unwrapped.context_id = context_id

        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

        while n < num_episodes:

            a = get_action(o)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1

            if d:
                logger.store(EpRet=ep_ret, EpLen=ep_len)

                assert context_id == info['context']
                obstacles = info['obstacles']
                goodies = info['goodies']
                goal = info['goal_reached']
                logger.store(**{f'EpRet_{context_id}': ep_ret,
                                f'EpLen_{context_id}': ep_len,
                                f'NObs_{context_id}': obstacles,
                                f'NGoodies_{context_id}': goodies,
                                f'GoalReached_{context_id}': goal})

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    for key in logger.epoch_dict.keys():
        if "EpRet_" in key and len(logger.epoch_dict[key]) > 0:
            logger.log_tabular(key, average_only=True)
            logger.log_tabular(key.replace("EpRet_", "EpLen_"), average_only=True)
            logger.log_tabular(key.replace("EpRet_", "NObs_"), average_only=True)
            logger.log_tabular(key.replace("EpRet_", "NGoodies_"), average_only=True)
            logger.log_tabular(key.replace("EpRet_", "GoalReached_"), average_only=True)

    stats = logger.log_current_row

    goals = {}
    obstacles = {}
    goodies = {}

    for ctx in range(n_contexts):

        try:
            goal = stats[f"GoalReached_{ctx}"]
            remaining_obs = stats[f"NObs_{ctx}"]
            remaining_goodies = stats[f"NGoodies_{ctx}"]

            goals[f'Score_{ctx}'] = goal
            obstacles[f'Score_{ctx}'] = remaining_obs
            goodies[f'Score_{ctx}'] = remaining_goodies

        except KeyError:
            warnings.warn(f'No stats for context {ctx} found! Try to run more episodes')

    obstacles = np.asarray(list(obstacles.values()))/env.unwrapped.n_obstacles
    goodies = np.asarray(list(goodies.values()))/env.unwrapped.n_goodies
    goals = np.asarray(list(goals.values()))

    score = evaluation_metric(goals, obstacles, goodies, w_G=0, w_OG=1)
    pair_score = score.reshape(-1, 2).min(1)

    avg_score = pair_score.mean()

    god_score = (pair_score >= 0.7).sum(0) / len(pair_score)

    log_writer.add_scalar('eval_stats/AvgEpRet', stats['AverageEpRet'], eval_step)
    log_writer.add_scalar('eval_stats/AvgEpLen', stats['EpLen'], eval_step)
    log_writer.add_scalar('eval_stats/AvgScore', avg_score, eval_step)
    log_writer.add_scalar('eval_stats/GOD_0.7', god_score, eval_step)
    for context_id in range(n_contexts):
        log_writer.add_scalar(f'eval_stats/context_{context_id}/EpRet', stats[f"EpRet_{context_id}"], eval_step)
        log_writer.add_scalar(f'eval_stats/context_{context_id}/EpLen', stats[f'EpLen_{context_id}'], eval_step)
        log_writer.add_scalar(f'eval_stats/context_{context_id}/NObs', stats[f'NObs_{context_id}'], eval_step)
        log_writer.add_scalar(f'eval_stats/context_{context_id}/NGoodies', stats[f'NGoodies_{context_id}'], eval_step)
        log_writer.add_scalar(f'eval_stats/context_{context_id}/GoalReached', stats[f'GoalReached_{context_id}'], eval_step)

    return god_score
