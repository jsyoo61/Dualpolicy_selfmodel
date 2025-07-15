
# assert args.ignore is None or args.independent is None, 'Only one of ignore or independent must be given. Received both.'
# %%
import itertools as it
import multiprocessing
import os
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import tools as T

# %%
if False:
    pass
    # %%
    # from tensorflow.python.summary.summary_iterator import summary_iterator
    os.chdir('..')
    # %%

    exp_dir = Path('exp/PlanAgentReproduce12')
    exp_dir = Path('exp/PlanAgentReproduce6')
    path_tb = exp_dir/'0/tensorboard/'
    path_tb_file = T.os.listdir(path_tb, join=True, isfile=True)[0]

    ev=EventAccumulator(path_tb_file)
    dir(ev)
    dir(ev.scalars)
    dir(ev.tensors)
    ev.Scalars('actorcritic/entropy_loss')
    ev.scalars.Items('worldmodel/loss_reward')
    list(ev.scalars)
    ev.scalars.Keys()
    ev.scalars.size
    it=summary_iterator(path_tb_file)
    l_summary = [summary for summary in summary_iterator(path_tb_file)]
    l_summary[:5]
    summary = l_summary[1]
    type(summary)
    ev.Tags()

    ev.Reload()
    loss = ev.scalars.Items('worldmodel/Loss')
    loss2 = ev.Scalars('worldmodel/Loss')
    ev.Tags()
    ev._ProcessScalar()

    # %%
    # Dummy class for debugging
    class Dummy():
        """ Dummy class for debugging """
        def __init__(self):
            pass
    # os.chdir('SelfModel')
    args = Dummy()
    args.exp_dir='exp/PlanAgentReproduce3'

    args.independent=None
    args.ignore='run.random_seed'
    # args.independent=['agent.buffer_size', 'agent.plan.probability', 'train.train_interval']
    os.chdir('..')

# %%
def duplicate_key(d, key):
    '''
    if "key" in d (dict), then return key1 or key2 ... until there's no duplicates.
    '''
    i=1
    key_ = dcopy(key)
    while key_ in d.keys():
        key_ = key+str(i)
        i+=1
    return key_

def walk(walk_dir, exp_dict={}):
    print(walk_dir, end='')
    exp_dict = exp_dict.copy()

    if len(T.os.listdir(walk_dir, isdir=True)) == 0:
        print('Why am I in an empty directory?')
    else:
        print('') # line break
    for subdir in T.os.listdir(walk_dir, join=True, isdir=True):
        if os.path.exists(os.path.join(subdir, '.hydra')):
            join_result(subdir, exp_dict)
        else:
            exp_dict_subdir = exp_dict.copy()
            exp_dict_subdir[duplicate_key(exp_dict_subdir, 'exp')] = os.path.basename(subdir)
            walk(subdir, exp_dict=exp_dict_subdir)

# %%
def overrides_to_dict(listconfig):
    d = {}
    for cfg in listconfig:
        key, value = cfg.split('=')
        d[key]=value
    return d

def load_cfg(subdir):
    subdir = Path(subdir)
    assert (subdir/'.hydra').exists(), f'the subdir ({subdir}) of args.exp_dir does not have ".hydra" directory'

    # Load full configuration
    cfg = OmegaConf.load(subdir/'.hydra/config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = T.unnest_dict(cfg)

    # Load overrides
    cfg_overrides = OmegaConf.load(Path(subdir)/'.hydra/overrides.yaml')
    cfg_overrides = overrides_to_dict(cfg_overrides)
    return cfg, cfg_overrides

def load_result_df(subdir, prefix='train'):
    # Load result xlsx
    path_file = Path(subdir)/'result'/f'{prefix}_result.xlsx'
    if path_file.exists():
        df = pd.read_excel(path_file, index_col=0)
    else:
        df = None
    return df

def load_action_type(subdir, prefix='train'):
    path_file = Path(subdir)/'result'/f'{prefix}_action_type.p'
    if path_file.exists():
        action_counter = T.load_pickle(path_file)
        action_map, action_history = action_counter.map, np.array(action_counter.sequence)
    else:
        action_map, action_history = {}, []
    return action_map, action_history

def load_tensorboard(subdir):
    path_tb = Path(subdir)/'tensorboard/'
    if path_tb.exists():
        l_path_tb_file = T.os.listdir(path_tb, join=True, isfile=True)

        if len(l_path_tb_file)!=1:
            print(f'valid tensorboard log file not found: {subdir}')
            d_tensorboard = {}
        else:
            path_tb_file = l_path_tb_file[0]
            event=EventAccumulator(path_tb_file)
            event.Reload()
            d_tensorboard = {}

            for tag_criterion, tag_value in event.Tags().items():
                if tag_criterion=='scalars':
                    for tag in tag_value:
                        values = event.Scalars(tag)
                        value_df = pd.DataFrame(values)
                        value_df.sort_values(by='step')

                        d_tensorboard[tag] = value_df

    else:
        d_tensorboard = {}

    return d_tensorboard

def get_missing_subdir(l_subdir, l_action_history, l_df, l_tensorboard):
    d_missing = {}

    # action_history
    l_action_history_missing = [subdir for subdir, action_history in zip(l_subdir, l_action_history) if len(action_history)==0]
    d_missing['action_history'] = l_action_history_missing

    # df
    l_df_missing = [subdir for subdir, df in zip(l_subdir, l_df) if df is None]
    d_missing['df'] = l_df_missing

    # tensorboard
    # l_tensorboard_missing = [subdir for subdir, tensorboard in zip(l_subdir, l_tensorboard) if len(tensorboard)==0]
    # d_missing['tensorboard'] = l_tensorboard_missing

    return d_missing

# %%
def retrieve_result(exp_dir, prefix='train'):
    print(f'Loading files... ({prefix})')
    l_subdir = [exp_dir_ for exp_dir_ in T.os.listdir(exp_dir, isdir=True, join=True) if os.path.basename(exp_dir_)[0]!='.'] # Exclude hidden directory

    with multiprocessing.Pool() as p:
        # functions with list(map()) are light functions, and p.map are heavy functions.
        # Light functions are faster on a single process, and heavy functions are faster on multiprocess
        print('Loading: config')
        l_cfg_overrides = list(map(load_cfg, tqdm(l_subdir)))
        l_cfg, l_overrides = [cfg_overrides[0] for cfg_overrides in l_cfg_overrides], [cfg_overrides[1] for cfg_overrides in l_cfg_overrides]

        print('Loading: action_map')
        # l_action_map_history = list(map(load_action_type, tqdm(l_subdir)))
        l_action_map_history = [load_action_type(subdir, prefix=prefix) for subdir in tqdm(l_subdir)]
        l_action_map, l_action_history = [action_map_history[0] for action_map_history in l_action_map_history], [action_map_history[1] for action_map_history in l_action_map_history]

        print('Loading: result')
        # l_df = list(map(load_result_df, tqdm(l_subdir)))
        l_df = [load_result_df(subdir, prefix=prefix) for subdir in tqdm(l_subdir)]

        # print('Loading: tensorboard')
        # l_tensorboard = p.map(load_tensorboard, tqdm(l_subdir))
        l_tensorboard=[]

        '''
        TODO: check multiple argument of list(map())
        load_action_type
        load_result_df
        '''

    df_cfg, df_overrides = pd.DataFrame(l_cfg), pd.DataFrame(l_overrides)
    d_missing = get_missing_subdir(l_subdir=l_subdir, l_action_history=l_action_history, l_df=l_df, l_tensorboard=l_tensorboard)
    print('[Missing]')
    for key, l_subdir_missing in d_missing.items():
        print(f'({key})\n'+'\n'.join(l_subdir_missing))

    return df_cfg, l_df, l_action_history, l_action_map, df_overrides

# %%
'''
move plot functions to plot.py
'''
# Plot functions

# Paramters for plotting
figsize=(10,10)
nrows, ncols = 1,2
alpha = 0.3
margin_ratio = 0.05

window_ratio = 0.05 # percent of total number of episodes (%)

def plot_cumsum(cum_reward, reward_type_map_inv, ax=None):
    mean, std = np.mean(cum_reward, axis=0), np.std(cum_reward, axis=0)

    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, (mean_, std_) in enumerate(zip(mean, std)):
        reward_type = reward_type_map_inv[i]
        ax.plot(mean_, label=reward_type)
        ax.fill_between(range(len(mean_)), y1=mean_-std_, y2=mean_+std_, alpha=alpha)

    legend = ax.legend()
    legend.set_title('Reward type')

    ax.set_ylabel('Cumulative reward count')
    ax.set_xlabel('Episode')
    ax.set_title('Cumulative reward per reward type')

    return ax

def plot_ratio(reward_ratio, reward_type_map_inv, ax=None):
    '''
    reward_ratio: shape of (n_experiments, n_reward_type, n_episode)
    '''
    mean, std, N = np.mean(reward_ratio, axis=0), np.std(reward_ratio, axis=0), len(reward_ratio)
    se = std/np.sqrt(N)
    shadow = std

    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, (mean_, shadow_) in enumerate(zip(mean, shadow)):
        reward_type = reward_type_map_inv[i]
        ax.plot(mean_, label=reward_type)
        ax.fill_between(range(len(mean_)), y1=mean_-shadow_, y2=mean_+shadow_, alpha=alpha)

    legend = ax.legend()
    legend.set_title('Type')
    ax.set_ylim(U.compute_margin(0, 1, margin_ratio))
    ax.set_ylabel('Ratio (%)')
    ax.set_xlabel('Episode')
    ax.set_title(f'Ratio with smoothing window ({int(window_ratio*100)}%)')

    return ax

def plot(df_overrides, l_df,  l_action_history, l_action_map, prefix=''):
    # Same number of episodes
    if T.equal([len(df) for df in l_df]):
        pass
    # Different number of episodes
    else:
        pass

    # Trimming rewards
    l_reward = [df['reward'].values for df in l_df]
    reward = np.stack(l_reward,axis=0)
    reward_types = sorted(np.unique(reward))
    reward_type_map = {reward_type:i for i, reward_type in enumerate(reward_types)}
    reward_type_map_inv = T.reverse_dict(reward_type_map)

    # Window for smoothing
    window = int(len(l_df[0]) * window_ratio)
    v=np.ones((window), dtype=int)

    # Cumulative reward plot per reward type vs episode
    n_experiment, n_episode = reward.shape[0], reward.shape[1]

    # Reward ratio
    reward_onehot = np.zeros((n_experiment, n_episode, len(reward_types)), dtype=bool)
    for i, reward_type in enumerate(reward_types):
        reward_onehot[(reward==reward_type), i] = True
    reward_onehot = np.transpose(reward_onehot, axes=(0,2,1)) # (n_experiment, reward types, n_episode)
    assert (reward_onehot.sum(1)==1).all()
    reward_windowcount = np.apply_along_axis(lambda x: np.convolve(x, v, mode='same'), axis=1, arr=reward_onehot.reshape(-1, n_episode)).reshape(n_experiment, len(reward_types), n_episode)
    reward_ratio = reward_windowcount / reward_windowcount.sum(1)[:,None,:] # Sum over reward type dimension
    # cum_reward = np.stack([np.cumsum(reward==reward_type, axis=1) for reward_type in reward_types], axis=1)

    # Action ratio plot
    l_n_actions = [len(action_history) for action_history in l_action_history]
    max_n_actions, min_n_actions = max(l_n_actions), min(l_n_actions)
    if min_n_actions>0:
        l_action_history_ratio = []
        for action_history, action_map in tqdm(zip(l_action_history, l_action_map), total=len(l_action_history)):
            action_history_onehot_ = np.zeros((len(action_map), len(action_history)), dtype=bool)
            for i, action_type in enumerate(action_map.values()):
                action_history_onehot_[i, action_history==action_type] = True

            window = int(len(action_history) * window_ratio)
            v=np.ones((window), dtype=int)
            action_history_windowcount = np.apply_along_axis(lambda x: np.convolve(x, v, mode='same'), axis=1, arr=action_history_onehot_)
            action_history_ratio = action_history_windowcount / action_history_windowcount.sum(0)[None,:]
            action_history_ratio = np.apply_along_axis(lambda x: np.interp(np.linspace(0,1,max_n_actions), np.linspace(0,1,len(x)), x), axis=1, arr=action_history_ratio)

            l_action_history_ratio.append(action_history_ratio)
        action_history_ratio = np.array(l_action_history_ratio)

    # plotting
    # result_summary
    print(f'Number of experiments for cfg_combination:')
    for crit_combination in it.product(*[df_overrides[crit].unique() for crit in args.independent]):
        if len(crit_combination)==0:
            i_crit=slice(None)
            print(f'[cfg: {crit_combination}][N: {len(df_overrides)}]')
        else:
            i_crit = [df_overrides[crit].values==crit_value for crit, crit_value in zip(args.independent, crit_combination)]
            i_crit = np.all(i_crit, axis=0)
            print(f'[cfg: {crit_combination}][N: {i_crit.sum()}]')

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        title = '\n'.join([f'{crit}={crit_value}' for crit, crit_value in zip(args.independent, crit_combination)])
        fig.suptitle(title)

        # episode vs reward ratio plot per reward type
        reward_ratio_ = reward_ratio[i_crit]
        plot_ratio(reward_ratio_, reward_type_map_inv=reward_type_map_inv, ax=axes[0])

        # episode vs action ratio plot per action type
        if min_n_actions>0:
            action_history_ratio_ = action_history_ratio[i_crit]
            l_action_map_ = [l_action_map[i] for i in np.where(i_crit)[0]]
            assert T.equal(l_action_map_)
            action_map = l_action_map_[0]
            action_map_inv = T.reverse_dict(action_map)

            plot_ratio(action_history_ratio_, reward_type_map_inv=action_map_inv, ax=axes[1])

        fig.savefig(path_save/f'{prefix}{crit_combination}.png')
        plt.close(fig)

# %%
'''
move metric functions to metric.py
'''
def summarize_scores(df_overrides, l_df, prefix=''):

    l_reward = [df['reward'].values for df in l_df]
    reward_types = sorted(np.unique(l_reward))

    # Reward early, mid, end


    # Reward all
    for reward_type in reward_types:
        df_overrides[f'reward.{reward_type}']=0

    for i, df in enumerate(l_df):
        reward_count_dict = df['reward'].value_counts().to_dict()

        for reward_type in reward_types:
            reward_count = reward_count_dict.get(reward_type, 0)
            reward_ratio = reward_count/len(df)
            df_overrides.loc[i,f'reward.{reward_type}']=reward_ratio

    df_overrides.to_excel(path_save/f'{prefix}summary.xlsx')

def trim_independent_variables(cfg):
    l_overrides = []
    l_invalid_dir = []
    for subdir in tqdm(T.os.listdir(cfg.exp_dir, isdir=True, join=True)):
        if os.path.basename(subdir)[0]=='.': # Hidden directory
            l_invalid_dir.append(subdir)
            continue
        subdir = Path(subdir)
        overrides = OmegaConf.load(subdir/'.hydra/overrides.yaml')
        l_overrides.append(overrides)
    df_overrides = pd.DataFrame(map(overrides_to_dict, l_overrides))

    independent_variables = {}
    for column in df_overrides.columns:
        try: # Unhashable values raise Error
            cfg_list = df_overrides[column].unique().tolist()
        except:
            cfg_list = df_overrides[column].astype(str).unique().tolist()

        if len(cfg_list) > 1:
            independent_variables[column] = cfg_list

    import pprint
    print('[Independent variables of this experiment]')
    pprint.pprint(independent_variables)

    del independent_variables[cfg.ignore]
    cfg.independent = list(independent_variables.keys())

    print(f'Independent variables for plotting: {cfg.independent}')
    return cfg

# %%
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--exp_dir', type=str, required=True, help='')
    parser.add_argument('--independent', type=str, nargs='+', help='independent variables')
    parser.add_argument('--ignore', type=str, default='run.random_seed', help='variables to ignore')
    parser.add_argument('--supress_error', type=bool, default=False, help='')
    args = parser.parse_args()

    # %%
    # Need this to run as "python analysis/test.py"
    import sys
    sys.path.append(os.getcwd())

    # %%
    import utils as U

    # %%
    print(f'[CWD: {os.getcwd()}][exp_dir: {args.exp_dir}]')
    print(f'criterions: {args.independent}')

    # %%
    if args.independent is None:
        l_overrides = []
        l_invalid_dir = []
        for subdir in tqdm(T.os.listdir(args.exp_dir, isdir=True, join=True)):
            if os.path.basename(subdir)[0]=='.': # Hidden directory
                l_invalid_dir.append(subdir)
                continue
            subdir = Path(subdir)
            overrides = OmegaConf.load(subdir/'.hydra/overrides.yaml')
            l_overrides.append(overrides)
        df_overrides = pd.DataFrame(map(overrides_to_dict, l_overrides))

        independent_variables = {}
        for column in df_overrides.columns:
            try: # Unhashable values raise Error
                cfg_list = df_overrides[column].unique().tolist()
            except:
                cfg_list = df_overrides[column].astype(str).unique().tolist()

            if len(cfg_list) > 1:
                independent_variables[column] = cfg_list

        import pprint
        print('[Independent variables of this experiment]')
        pprint.pprint(independent_variables)

        del independent_variables[args.ignore]
        args.independent = list(independent_variables.keys())

    if args.ignore is not None:
        print(f'Independent variables for plotting: {args.independent}')
        path_save = Path(f'analysis/{args.exp_dir}')

        # Plot training result
        os.makedirs(path_save/'train', exist_ok=True)
        df_cfg, l_df, l_action_history, l_action_map, df_overrides = retrieve_result(args.exp_dir, 'train')
        plot(df_overrides, l_df, l_action_history, l_action_map, prefix='train/')
        summarize_scores(df_overrides, l_df, 'train/')

        # Plot testing result
        os.makedirs(path_save/'test', exist_ok=True)
        df_cfg, l_df, l_action_history, l_action_map, df_overrides = retrieve_result(args.exp_dir, 'test')
        plot(df_overrides, l_df, l_action_history, l_action_map, prefix='test/')
        summarize_scores(df_overrides, l_df, 'test/')
