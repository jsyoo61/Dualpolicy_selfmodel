# %%
import itertools as it
import logging
import os
from pathlib import Path
from tqdm import tqdm

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd

print(f'[CWD: {os.getcwd()}]')
import sys
sys.path.append(os.getcwd())  # Ensure the root directory is in the path to import "analysis" directory as a package

import analysis.test as test
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
        'exp_name=assumption6',
    ]
    cfg = hydra.compose(config_name='plot', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))

# %%
log = logging.getLogger(__name__)

# %%
def plot(reward_ratio, df_overrides, path_save, cfg):
    # plotting
    l_title = ['Cumulative_ratio_of_death', 'Cumulative_ratio_of_timeout', 'Cumulative_ratio_of_success']
    l_ylabel = ['Death (%)', 'Timeout (%)', 'Success (%)']
    l_reward_ratio_type = [reward_ratio_ for reward_ratio_ in np.transpose(reward_ratio, axes=(1,0,2))]

    n=None

    for i, reward_ratio_type in enumerate(l_reward_ratio_type):
        reward_ratio_type.shape
        fig, ax = plt.subplots(nrows=cfg.plot.nrows, ncols=cfg.plot.ncols, figsize=(cfg.plot.figsize[0]*cfg.plot.ncols, cfg.plot.figsize[1]*cfg.plot.nrows))

        crit_list = [df_overrides[crit].unique() for crit in cfg.independent]
        agent_list = crit_list[cfg.independent.index('agent')]
        agent_list_ordered = sorted(agent_list, key=lambda x:agent_order.index(x))
        crit_list[0] = agent_list_ordered

        for crit_combination in it.product(*crit_list):
            print(crit_combination)
            i_crit = S.where_crit(crit_combination, df_overrides, cfg)
            reward_ratio_type_agent = reward_ratio_type[i_crit]

            i_samples = slice(None,n)
            reward_ratio_type_agent_ = reward_ratio_type_agent[i_samples]
            y = reward_ratio_type_agent_.mean(0)
            # dy = reward_ratio_type_agent_.std(0)
            if cfg.shade == 'std':
                dy = reward_ratio_type_agent.std(0)
            elif cfg.shade == 'CI':
                dy = 1.96 * reward_ratio_type_agent_.std(0)/np.sqrt(len(reward_ratio_type_agent_)) # 95% Confidence interval
            else:
                raise Exception(f'Undefined shade: {cfg.shade}')

            S.plot_line(y=y,dy=dy, ax=ax, alpha=cfg.plot.alpha)

        legend_objects = [(line, shade) for line, shade in zip(ax.lines, ax.collections)]
        ax.legend(legend_objects, cfg.legend)
        ax.set_ylim(U.compute_margin(0, 1, cfg.plot.margin_ratio))
        ylabel = l_ylabel[i]
        # ax.set_ylabel('Ratio (%)')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Training episode')
        title = l_title[i]
        # ax.set_title(title)

        fig.savefig(path_save/(title+'.png'), bbox_inches='tight')

d_legend = {
'assumption6': ['Simple Agent', 'Shared Policy Agent', 'Dual Policy Agent'],
'speed2': ['Simple Agent', 'Shared Policy Agent', 'Dual Policy Agent'],
'mapsize3': ['Simple Agent', 'Shared Policy Agent', 'Dual Policy Agent'],
'assumption_innatetrigger4': ['Simple Agent', 'Shared Policy Agent', 'Dual Policy Agent'],
}
d_shade = {
'assumption6': 'std',
'mapsize3': 'std'
}
agent_order = ['Simple', 'Plan', 'PlanSelf', 'Innate', 'PlanInnate', 'PlanInnateTrigger', 'PlanInnateSelf', 'PlanSelfPolicy', 'PlanSelfValue']

# %%
@hydra.main(config_path=os.path.join(os.getcwd(),'config'), config_name='plot', version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    path_save, cfg = S.setup(cfg, save_dir='training')

    # Plot training result
    df_cfg, l_df, l_action_history, l_action_map, df_overrides = test.retrieve_result(cfg.exp_dir, 'train')
    
    # Trimming rewards
    l_reward = [df['reward'].values for df in l_df]
    reward = np.stack(l_reward,axis=0)
    reward_types = sorted(np.unique(reward))
    reward_type_map = {reward_type:i for i, reward_type in enumerate(reward_types)}
    reward_type_map_inv = T.reverse_dict(reward_type_map)

    # Window for smoothing
    window = int(len(l_df[0]) * cfg.plot.window_ratio)
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

    cfg.legend = d_legend[cfg.exp_name]
    cfg.shade = d_shade.get(cfg.exp_name, 'CI') # Default: CI

    # %%
    if cfg.exp_name == 'assumption_innatetrigger4':
        # Exclude innate trigger
        index = df_overrides['agent']!='PlanInnateTrigger'

        df_overrides_, reward_ratio_ = df_overrides[index], reward_ratio[index]
    else:
        df_overrides_, reward_ratio_ = df_overrides, reward_ratio

    # %%
    if len(cfg.independent) > 1: # We're comparing agents, so independent variables other than "agent" are experiment sets
        exp_variables = [crit for crit in cfg.independent if crit != 'agent']
        for crit_combination in it.product(*[df_overrides_[crit].unique() for crit in exp_variables]):
            exp_name = ' '.join([f'{crit}_{crit_value}' for crit, crit_value in zip(exp_variables, crit_combination)])
            print(exp_name)

            indices = [(df_overrides_[crit]==crit_value).values for crit, crit_value in zip(exp_variables, crit_combination)]
            indices = np.all(indices, axis=0)

            reward_ratio__ = reward_ratio_[indices]
            df_overrides__ = df_overrides_[indices]
            path_save_ = path_save / exp_name
            os.makedirs(path_save_, exist_ok=True)
            plot(reward_ratio__, df_overrides__, path_save_, cfg)
    else:
        plot(reward_ratio_, df_overrides_, path_save, cfg)

# %%
if __name__=='__main__':
    main()

# %%
if False:
    pass
    # %%
    os.chdir('scripts/figures')
    os.getcwd()
    
# %%
