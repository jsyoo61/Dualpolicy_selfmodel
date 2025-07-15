# %%
import itertools as it
import os
from pathlib import Path
from tqdm import tqdm

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import stable_baselines3

print(f'[CWD: {os.getcwd()}]')
import sys
sys.path.append(os.getcwd())  # Ensure the root directory is in the path to import "analysis" directory as a package

import analysis.test as test
import agent_models
import run
import agents
import tools as T
import utils as U
import scripts.figures.setting as S

# %%
if False:
    if os.path.basename(os.getcwd())!='code':
        os.chdir('../..')
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(os.getcwd(), 'config'), job_name='debug')
    overrides = [
        'exp_name=cost4',
    ]
    cfg = hydra.compose(config_name='plot', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))

# %%
# cfg.exp_name = 'cost'
# cfg.exp_name = 'cost2'
# cfg.exp_name = 'cost3'
# cfg.exp_name = 'cost4'
# cfg.exp_name = 'cost_selfmodeldouble'
# cfg.exp_name = 'cost_innate'
# cfg.exp_name = 'cost_innate_selfmodeldouble'

def sum_of_hiddenlayers(index):
    return [sum(eval(i)) for i in index]

def rank_of_hiddenlayers(index):
    # rank = {'S': 0, 'M': 1, 'L': 2}
    rank = {'[32]': 0, '[64,64]': 1, '[128,128,128,128]': 2}
    return [rank[i] for i in index]

def nest_dict(d, format='.'):
    '''
    Convert a 1st order dictionary with format-separated-keys into a nested dictionary.
    '''
    n_d = {}
    for k, v in d.items():
        keys = k.split(format)
        d_ = n_d # Start from root
        for key in keys[:-1]: # Traverse to the last key
            if key not in d_: # If key does not exist, create a new dict
                d_[key] = {}
            d_ = d_[key] # Move to the next dict
        d_[keys[-1]] = v # Set the value to the last key
    return n_d

def get_param(dict_exp):
    dict_exp = nest_dict(dict_exp, format='.')
    cfg_exp = DictConfig(dict_exp)
    TrainerType = eval(cfg_exp.agent.actorcritic.TrainerType._target_)
    AgentClass = eval(cfg_exp.AgentClass._target_)
    env = hydra.utils.instantiate(cfg_exp.env) # No rendering during training, cuz we need lots of experience
    nn_actorcritic, model_actorcritic, trainer_actorcritic = agent_models.hack_stablebaselines(TrainerType=TrainerType, env=env, cfg=cfg_exp, tensorboard_writer=None)
    n_param = sum([np.prod(p.data.numpy().shape) for p in nn_actorcritic.parameters()])

    return n_param

# %%
@hydra.main(config_path=os.path.join(os.getcwd(),'config'), config_name='plot', version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    path_save, cfg = S.setup(cfg, save_dir='cost')
    cfg.plot.markersize=10

    # Plot test result
    df_cfg, l_df, l_action_history, l_action_map, df_overrides = test.retrieve_result(cfg.exp_dir, 'test')

    # Trimming rewards
    l_reward = [df['reward'].values for df in l_df]
    reward = np.stack(l_reward,axis=0)
    reward_types = sorted(np.unique(reward))
    reward_type_map = {reward_type:i for i, reward_type in enumerate(reward_types)}
    reward_type_map_inv = T.reverse_dict(reward_type_map)

    # Cumulative reward plot per reward type vs episode
    n_experiment, n_episode = reward.shape[0], reward.shape[1]

    # Reward ratio
    reward_onehot = np.zeros((n_experiment, n_episode, len(reward_types)), dtype=bool)
    for i, reward_type in enumerate(reward_types):
        reward_onehot[(reward==reward_type), i] = True
    reward_onehot = np.transpose(reward_onehot, axes=(0,2,1)) # (n_experiment, reward types, n_episode)
    assert (reward_onehot.sum(1)==1).all()

    reward_onehot.shape
    reward_ratio = reward_onehot.sum(-1) / reward_onehot.shape[-1]

    # plotting
    l_title = ['Ratio of death', 'Ratio of timeout', 'Ratio of success']
    l_ylabel = ['Death (%)', 'Timeout (%)', 'Success (%)']

    l_reward_ratio_type = [reward_ratio_ for reward_ratio_ in np.transpose(reward_ratio, axes=(1,0))]
    df_n_param = df_cfg.apply(lambda x: get_param(x.to_dict()), axis=1)
    x = df_n_param.unique() # np.array([549, 9413, 101253])

    for i, reward_ratio_type in enumerate(l_reward_ratio_type):
        fig, ax = plt.subplots(nrows=cfg.plot.nrows, ncols=cfg.plot.ncols, figsize=(cfg.plot.figsize[0]*cfg.plot.ncols, cfg.plot.figsize[1]*cfg.plot.nrows))

        # for layer_actorcritic in df_overrides['agent.actorcritic.layers'].unique(): # xaxis
        for layer_selfmodel in df_overrides['agent.selfmodel.NNClass.n_hidden_list'].unique(): # legend
            reward_ratio_type_ = reward_ratio_type[layer_selfmodel==df_overrides['agent.selfmodel.NNClass.n_hidden_list']]
            df_overrides_ = df_overrides[layer_selfmodel==df_overrides['agent.selfmodel.NNClass.n_hidden_list']].copy()
            df_overrides_['Performance'] = reward_ratio_type_
            df_overrides_
            gb = df_overrides_.groupby('agent.actorcritic.layers')
            mean, std, N = gb['Performance'].mean(), gb['Performance'].std(), gb['Performance'].count().values
            mean.sort_index(key=rank_of_hiddenlayers, inplace=True), std.sort_index(key=rank_of_hiddenlayers, inplace=True)
            # mean, std = mean.values.squeeze(1), std.values.squeeze(1)
            mean, std = mean.values, std.values
            yerr = 1.96 * std/np.sqrt(N) # 95% Confidence interval

            # ax.errorbar(x=x, y=mean, yerr=std, fmt='o', label=layer_selfmodel)
            ax.errorbar(x=x, y=mean, yerr=yerr, fmt='o-', markersize=10, capsize=15, label=layer_selfmodel)

        legend_objects = [(line, shade) for line, shade in zip(ax.lines, ax.collections)]
        ax.legend(title='Distilled policy size')
        ax.set_ylim(U.compute_margin(0, 1, cfg.plot.margin_ratio))
        ylabel = l_ylabel[i]

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Model-free policy parameters')
        ax.set_xscale('log')

        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0]*cfg.plot.margin_ratio, xlim[-1]/cfg.plot.margin_ratio)

        ax.set_xticks(x)
        # ax.set_xticklabels(df_overrides['agent.actorcritic.layers'].unique())
        title = l_title[i]
        # ax.set_title(title)

        fig.savefig(path_save/(title+'.png'), bbox_inches='tight')

if __name__=='__main__':
    main()