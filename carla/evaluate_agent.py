import cv2
import os
import warnings

import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import pandas as pd
import seaborn as sns
from carla.agents.utils.pytorch_utils import dict_obs_to_tensor, merge_dict_obs_list, merge_list_of_dicts

from carla.agents.utils.logx import EpochLogger
from carla.agents.eval_utils import evaluation_metric

from carla.train_agent import env_fnc
from carla.agents.utils.subproc_vec_env import SubprocVecEnv
from carla.agents.utils.dummy_vec_env import DummyVecEnv


def tsplot(ax, x, data, max_plot, **kw):

    est = np.mean(data, axis=0)

    if max_plot:
        cis = (np.min(data, axis=0), np.max(data, axis=0))
    else:
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)

    ax.fill_between(x, cis[0], cis[1], alpha=0.2)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def print_avg_cp_stats(final_dict, legend, order):
    # print average context pair statistics in latex table format
    measures = final_dict.keys()

    df = pd.DataFrame(columns=['exp_name', 'context_pair', 'run_id', 'measure_name', 'measure_value'])
    counter = 0
    for measure in measures:
        final_scores = final_dict[measure]

        for j, exp_name in enumerate(final_scores):
            context_scores = np.array(final_scores[exp_name])
            n_contexts = context_scores.shape[0]
            n_runs = context_scores.shape[1]

            # reshape context x seeds -> context_pairs x 2 x seeds
            score_pair = context_scores.reshape(-1, 2, context_scores.shape[-1])
            if "Goodie" in measure:
                # max over context pairs
                score_pair = score_pair.max(1)
            elif 'Goal' in measure or 'Obstacle' in measure:
                # min over context pairs
                score_pair = score_pair.min(1)
            else:
                raise NotImplementedError

            for cp in range(n_contexts//2):
                for r in range(n_runs):
                    df.loc[counter] = pd.Series(
                        {'exp_name': legend[exp_name], 'context_pair': cp, 'run_id': r, 'measure_name': measure,
                         'measure_value': score_pair[cp, r]})
                    counter += 1

    print('Average Aggregated Results')
    df_agg = df.groupby(by=["exp_name", 'measure_name']).agg({'measure_value': ['mean', 'std']}).reset_index()

    print(order)
    for lgd in legend.values():
        print(f"& {lgd}", end=' ')
        for m in order:
            value = df_agg[(df_agg['exp_name'] == lgd) & (df_agg['measure_name'] == m)]
            rmean = np.round(value.measure_value['mean'].to_list()[0], 3)
            rstd = np.round(value.measure_value['std'].to_list()[0], 3)
            print(f"& {rmean:1.3f} $\pm$ {rstd:1.3f}", end=" ")
        print("\\\\")


def plot_pair_scores_avgseed(final_dict, out_path, legend, title='Score', save_plot=False, w_G=0.5,
                             w_OG=0.5, max_plot=False, exp_id='default'):
    measures = final_dict.keys()

    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)

    color_palette = sns.color_palette("tab10", n_colors=10)

    score_pair_stats = {}
    for measure in measures:
        final_scores = final_dict[measure]

        for j, exp_name in enumerate(final_scores):

            if exp_name not in score_pair_stats:
                score_pair_stats[exp_name] = {}

            s = np.asarray(final_scores[exp_name])
            score_pair_stats[exp_name][measure] = s

    linestyles = ['-', '--', '-.', ':']
    thresholds = np.arange(0, 1, 0.01)
    for j, exp_name in enumerate(score_pair_stats):

        goodies = score_pair_stats[exp_name]['Remaining Goodie']
        obstacles = score_pair_stats[exp_name]['Remaining Obstacle']
        goal = score_pair_stats[exp_name]['Goal Reached']

        d = []

        pair_score = evaluation_metric(goal, obstacles, goodies, w_G=w_G, w_OG=w_OG)


        pair_score = pair_score.reshape(-1, 2, pair_score.shape[-1])
        pair_score = pair_score.min(1)

        for i in thresholds:
            s = (pair_score >= i).sum(0) / len(pair_score)

            d.append(s)

        d = np.stack(d).T

        tsplot(ax, thresholds, d, max_plot, label=legend[exp_name], linewidth=3,
               linestyle=linestyles[min(j//len(color_palette), len(linestyles))],
               color=color_palette[j%len(color_palette)])

    fontsize = 30
    if w_OG == 0 and w_G == 1:
        plt.title(title + ' GR-PCP vs Threshold', fontsize=fontsize)
        y_label = "GR-PCP"
    elif w_OG == 1 and w_G == 0:
        plt.title(title + ' GOD-PCP vs Threshold', fontsize=fontsize)
        y_label = "GOD-PCP"
    elif w_OG == 0.5 and w_G == 0.5:
        plt.title(title + ' Average-SCPR vs Threshold', fontsize=fontsize)
        y_label = "Average-SCPR"
    else:
        plt.title(title + ' SCPR vs Threshold ($w_G$:' + str(w_G) + ',$w_{OG}$:' + str(w_OG) + ')', fontsize=fontsize)
        y_label = "SCPR"

    plt.ylim((-0.05, 1.1))
    ax.set_xlabel("Threshold", fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    lgd = plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0., fontsize=fontsize)

    if save_plot:
        if not os.path.exists(os.path.join(out_path, 'plots')):
            os.makedirs(os.path.join(out_path, 'plots'))
        plt.savefig(os.path.join(out_path, 'plots', "{}_pair_Score_wG{}_wOG{}.png".format(exp_id, w_G, w_OG)),
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def eval_policy(env, get_action, num_episodes=100, n_contexts=6, render=False, store_video=False):

    logger = EpochLogger()

    is_dummy_vec_env = isinstance(env, DummyVecEnv)

    if is_dummy_vec_env:
        env.envs[0].unwrapped.random_context = False
    else:
        for worker in env.remotes:
            worker.send(('set_random_context', False))

    observations = []

    for context_id in range(n_contexts):

        if is_dummy_vec_env:
            env.envs[0].unwrapped.context_id = context_id
        else:
            for worker in env.remotes:
                worker.send(('set_context_id', context_id))

        o, ep_ret, ep_len = env.reset(), np.zeros(env.num_envs), np.zeros(env.num_envs, dtype=np.int32)
        n = 0

        while n < num_episodes:

            if is_dummy_vec_env and render:
                img = env.render()
                time.sleep(1e-3)

                if store_video:
                    observations.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            a = get_action(o)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            info = merge_list_of_dicts(info)

            for proc_idx in range(env.num_envs):
                terminal = d[proc_idx]

                if terminal:
                    logger.store(EpRet=ep_ret[proc_idx], EpLen=ep_len[proc_idx])

                    if is_dummy_vec_env:
                        print('Episode %d \t Context %d \t EpRet %.3f \t EpLen %d' % (n, context_id, ep_ret[proc_idx], ep_len[proc_idx]))

                    assert context_id == info['context'][proc_idx]
                    obstacles = info['obstacles'][proc_idx]
                    goodies = info['goodies'][proc_idx]
                    goal = info['goal_reached'][proc_idx]
                    logger.store(**{f'EpRet_{context_id}': ep_ret[proc_idx],
                                    f'EpLen_{context_id}': ep_len[proc_idx],
                                    f'NObs_{context_id}': obstacles,
                                    f'NGoodies_{context_id}': goodies,
                                    f'GoalReached_{context_id}': goal})

                    ep_ret[proc_idx] = 0
                    ep_len[proc_idx] = 0

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

    # return scores, goals, obstacles, goodies, stats
    return goals, obstacles, goodies, stats, observations


def load_pytorch_policy(fpath, itr, observation_space):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = os.path.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname, map_location='cuda')

    model.eval()

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = dict_obs_to_tensor(merge_dict_obs_list(x, observation_space), torch.device('cuda'))
            action = model.act(x)
        return action

    return get_action, model


def load_policy_and_env(fpath, context_config=None, n_proc=1, itr='last'):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    corrected_itr = itr

    if corrected_itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        pytsave_path = os.path.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if
                 'best' not in x and len(x) > 8 and 'model' in x]
        corrected_itr = '%d' % max(saves) if len(saves) > 0 else ''

    with open(os.path.join(fpath, 'config.json')) as json_file:
        args = json.load(json_file)
        env_kwargs = args['env_kwargs']

        if context_config is not None:
            env_kwargs['context_config'] = context_config

        env_fns = [env_fnc(rank=i, **env_kwargs) for i in range(n_proc)]

        if n_proc == 1:
            env = DummyVecEnv(env_fns)
        else:
            env = SubprocVecEnv(env_fns)

    get_action, model = load_pytorch_policy(fpath, corrected_itr, env.observation_space)

    return env, get_action, model, corrected_itr


def plot_return(data, legend, out_path, title, exp_id):
    fontsize = 30

    color_palette = sns.color_palette("tab10", n_colors=10)
    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)

    for i, key in enumerate(data):
        tsplot(ax, np.arange(data[key].shape[1]), data[key], max_plot=False, label=legend[key], linewidth=3,
               color=color_palette[i%len(color_palette)])

    ax.set_xlabel("Epochs", fontsize=fontsize)
    ax.set_ylabel("Cumulative Reward", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    lgd = plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0., fontsize=fontsize)
    plt.title(title + " Cumulative Reward", fontsize=fontsize)

    plt.savefig(os.path.join(out_path, 'plots', "{}_cumulative_reward.png".format(exp_id)),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--episodes', '-n', help="number of episodes per context", type=int, default=100)
    parser.add_argument('--itr', '-i', type=str, default="last")
    parser.add_argument('--save_plot', default=False, action='store_true')
    parser.add_argument('--recalc', default=False, action='store_true')
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--plot_dir', type=str)
    parser.add_argument("--context_config", required=True, help="which context configuration to load")
    parser.add_argument('--seeds', nargs='+', default=[])
    parser.add_argument('--legend', nargs='+')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--w_G', type=float, default=0.5)
    parser.add_argument('--w_OG', type=float, default=0.5)
    parser.add_argument('--n_proc', type=int, default=8)
    parser.add_argument('--exp-id', type=str, default='default')
    parser.add_argument('--max_plot', default=False, action='store_true')
    parser.add_argument('--plot_return', default=False, action='store_true')

    args = parser.parse_args()

    print('args:', args)

    goals = {}
    obstacles = {}
    goodies = {}

    return_over_epochs = {}

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    n_contexts = None

    legend = {}

    n_obstacles = None
    n_goodies = None

    config_name = os.path.basename(args.context_config).split(".")[0]

    for idx, fpath in enumerate(args.logdir):
        exp_name = os.path.basename(fpath)
        legend[exp_name] = args.legend[idx]
        return_over_epochs[exp_name] = []

        if args.result_dir != '':
            out_path = args.result_dir
        else:
            out_path = fpath

        goals[exp_name] = {}
        obstacles[exp_name] = {}
        goodies[exp_name] = {}

        if len(args.seeds) == 0:
            listed_dirs = os.listdir(fpath)
        else:
            listed_dirs = []
            for s in args.seeds:
                drs = fpath.split('/')
                if drs[-1] == '':
                    dir_name = drs[-2]
                else:
                    dir_name = drs[-1]
                listed_dirs.append(os.path.join(fpath, dir_name + '_s{}'.format(s)))

        for i, run_path in enumerate(listed_dirs):
            exp_folder = os.path.join(exp_name, os.path.basename(run_path))
            avg_return = pd.read_csv(os.path.join(run_path, 'progress.txt'), sep="\t", usecols=['AverageEpRet'])
            return_over_epochs[exp_name].append(avg_return)

            env, get_action, model, corrected_itr = load_policy_and_env(run_path, args.context_config, args.n_proc, args.itr)

            if isinstance(env, DummyVecEnv):
                assert len(env.envs) == 1
                if n_contexts is None:

                    n_contexts = env.envs[0].unwrapped.n_contexts
                else:
                    # make sure to only evalute models with a same number of context_configurations
                    assert n_contexts == env.envs[0].unwrapped.n_contexts

                n_obstacles = env.envs[0].unwrapped.n_obstacles
                n_goodies = env.envs[0].unwrapped.n_goodies

            else:
                env.remotes[0].send(('n_contexts', None))
                if n_contexts is None:

                    n_contexts = env.remotes[0].recv()
                else:
                    # make sure to only evalute models with a same number of context_configurations
                    assert n_contexts == env.remotes[0].recv()

                # n_objects = env.unwrapped.n_goodies + env.unwrapped.n_obstacles
                env.remotes[0].send(('n_obstacles', None))
                n_obstacles = env.remotes[0].recv()

                env.remotes[0].send(('n_goodies', None))
                n_goodies = env.remotes[0].recv()

            if not os.path.exists(os.path.join(out_path, exp_folder)):
                os.makedirs(os.path.join(out_path, exp_folder))

            stat_file = os.path.join(out_path, exp_folder, f'evaluation_stat_{args.episodes}_{corrected_itr}_{config_name}.npy')
            goal_file = os.path.join(out_path, exp_folder, f'evaluation_goal_{args.episodes}_{corrected_itr}_{config_name}.npy')
            obstacle_file = os.path.join(out_path, exp_folder, f'evaluation_obstacle_{args.episodes}_{corrected_itr}_{config_name}.npy')
            goodie_file = os.path.join(out_path, exp_folder, f'evaluation_goodie_{args.episodes}_{corrected_itr}_{config_name}.npy')

            if not args.recalc and os.path.exists(goal_file):
                print(f'Loading scores for {exp_folder} with {args.episodes} episodes per context...')
                goal = np.load(goal_file, allow_pickle=True).item()
                obstacle = np.load(obstacle_file, allow_pickle=True).item()
                goodie = np.load(goodie_file, allow_pickle=True).item()
            else:
                start = time.time()
                print(f'Evaluating {exp_folder} for {args.episodes} episodes per context...')
                goal, obstacle, goodie, stat, _ = eval_policy(env, get_action, args.episodes, n_contexts)
                # time it
                elapsed = time.time() - start
                print('took {}\n'.format(time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))))

                np.save(stat_file, stat)
                np.save(goal_file, goal)
                np.save(obstacle_file, obstacle)
                np.save(goodie_file, goodie)

            for key in goal:
                if key in goals[exp_name]:
                    goals[exp_name][key].append(goal[key])
                else:
                    goals[exp_name][key] = [goal[key]]

                if key in goodies[exp_name]:
                    goodies[exp_name][key].append(goodie[key])
                else:
                    goodies[exp_name][key] = [goodie[key]]

                if key in obstacles[exp_name]:
                    obstacles[exp_name][key].append(obstacle[key])
                else:
                    obstacles[exp_name][key] = [obstacle[key]]

            env.close()

    final_goodies = {}
    final_obstacles = {}
    final_goals = {}

    for exp_name in goodies:
        final_goodies[exp_name] = []
        final_obstacles[exp_name] = []
        final_goals[exp_name] = []
        for key in goodies[exp_name]:
            final_goodies[exp_name].append(goodies[exp_name][key])
            final_goals[exp_name].append(goals[exp_name][key])
            final_obstacles[exp_name].append(obstacles[exp_name][key])

        final_goodies[exp_name] = np.asarray(final_goodies[exp_name]) / n_goodies
        final_obstacles[exp_name] = np.asarray(final_obstacles[exp_name]) / n_obstacles
        if args.plot_return:
            return_over_epochs[exp_name] = np.stack(return_over_epochs[exp_name])[..., 0]

    plot_pair_scores_avgseed({'Remaining Goodie': final_goodies,
                              'Goal Reached': final_goals,
                              'Remaining Obstacle': final_obstacles},
                             args.plot_dir, legend, title=args.title, save_plot=args.save_plot,
                             w_G=args.w_G, w_OG=args.w_OG, max_plot=args.max_plot, exp_id=args.exp_id)

    if args.plot_return:
        plot_return(return_over_epochs, legend, args.plot_dir, args.title, exp_id=args.exp_id)

    print_avg_cp_stats({'Remaining Goodie': final_goodies, 'Goal Reached': final_goals,
                        'Remaining Obstacle': final_obstacles}, legend,
                       ['Remaining Goodie', 'Remaining Obstacle', 'Goal Reached'])

