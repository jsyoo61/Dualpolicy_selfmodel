import os
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt

import analysis.test as test

# %%
def setup(cfg, save_dir=None):
    OmegaConf.set_struct(cfg, False)  # Make cfg writable
    cfg.exp_dir = f'exp/{cfg.exp_name}'
    cfg.ignore = 'run.random_seed'
    cfg.save_dir = f'figures/{cfg.exp_name}/{save_dir}'
    path_save = Path(cfg.save_dir)
    os.makedirs(path_save, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))
    plt.rcParams.update({'font.size': cfg.plot.fontsize})
    plt.rcParams.update({'font.family': cfg.plot.fontname})

    print(f'[CWD: {os.getcwd()}][exp_dir: {cfg.exp_dir}]')
    cfg = test.trim_independent_variables(cfg)

    return path_save, cfg

def plot_line(y,dy, ax=None, figsize=None, alpha=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(y)
    ax.fill_between(range(len(y)), y1=y-dy, y2=y+dy, alpha=alpha)

def where_crit(crit_combination, df_overrides, cfg):
    if len(crit_combination)==0:
        i_crit=slice(None)
        print(f'[cfg: {crit_combination}][N: {len(df_overrides)}]')
    else:
        i_crit = [df_overrides[crit].values==crit_value for crit, crit_value in zip(cfg.independent, crit_combination)]
        i_crit = np.all(i_crit, axis=0)
        print(f'[cfg: {crit_combination}][N: {i_crit.sum()}]')
    return i_crit
