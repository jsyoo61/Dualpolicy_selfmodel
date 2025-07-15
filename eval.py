from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils as U

# %%
def evaluate(df, cfg, agent, save_dir='result', prefix=''):
    """
    Evaluate the experiment results

    Parameters
    ----------
    df : pandas.DataFrame
        The following columns must exist: ['episode', 'reward', 'step']

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Explanation variable
    """
    # TODO: Might have to set dpi=300 for publication later.

    # Define directory to store all evaluation results
    p = Path(save_dir)
    save = save_dir is not None

    nrows=3
    ncols=2
    margin_ratio=0.05 # The margin should be n%
    columns = ['episode', 'reward', 'step']
    assert set(df.columns).issuperset(columns), f'result dataframe must contain the columns: {columns}, missing: {sorted(set(columns) - set(df.columns))}'

    def compute_margin(ymin, ymax, margin_ratio):
        margin = margin_ratio / (1-margin_ratio*2) * (ymax - ymin)
        return (ymin-margin, ymax+margin)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*cfg.eval.figsize[1], nrows*cfg.eval.figsize[0]))
    axes = axes.flatten()

    # episode vs reward curve
    df.plot(x='episode', y='reward', style='x-', ax=axes[0])
    axes[0].set_title('Reward per episode')
    axes[0].set_ylim(compute_margin(cfg.eval.learning_curve.reward_min, cfg.eval.learning_curve.reward_max, margin_ratio))
    axes[0].set_ylabel('Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend([])

    # episode vs step curve
    df.plot(x='episode', y='step', style='x-', ax=axes[1])
    margin = margin_ratio / (1-margin_ratio*2) * (cfg.eval.learning_curve.reward_max-cfg.eval.learning_curve.reward_min)
    axes[1].set_title('Length (# of step) of episode')
    axes[1].set_ylim(compute_margin(0, cfg.env.max_episode_steps, margin_ratio))
    axes[1].set_ylabel('Step')
    axes[1].set_xlabel('Episode')
    axes[1].legend([])


    # Cumulative reward plot per reward type vs episode
    if df['reward'].nunique() < 10: # Applies when the reward is categorical. Assume categorical when there's less than 10 types of sum(reward) from an episode
        for reward_type in sorted(df['reward'].unique()):
            axes[2].plot(np.cumsum(df['reward']==reward_type), label=reward_type)
        legend = axes[2].legend()
        legend.set_title('Reward type')

        axes[2].set_title('Cumulative reward per reward type')
        axes[2].set_ylabel('Cumulative count')
        axes[2].set_xlabel('Episode')

    # Reward ratio plot per reward type vs episode
        window_ratio = 0.05 # percent of total number of episodes (%)
        window = int(len(df) * window_ratio)
        rewards_onehot = np.zeros((len(df), df['reward'].nunique()), dtype=bool) # (number of episodes, reward types)
        for i, reward_type in enumerate(sorted(df['reward'].unique())):
            rewards_onehot[(df['reward'].values==reward_type), i] = True
        assert (rewards_onehot.sum(1)==1).all()
        v=np.ones((window), dtype=int)
        rewards_windowcount = np.stack([np.convolve(reward_type_onehot,v, mode='same') for reward_type_onehot in rewards_onehot.T], axis=1)
        rewards_ratio = rewards_windowcount / rewards_windowcount.sum(1)[...,None]

        for reward_type_ratio, reward_type in zip(rewards_ratio.T, sorted(df['reward'].unique())):
            axes[3].plot(reward_type_ratio, label=reward_type)
        legend = axes[3].legend()
        legend.set_title('Reward type')
        axes[3].set_title('Reward ratio within episode window')
        axes[3].set_ylim(compute_margin(0, 1, margin_ratio))
        axes[3].set_ylabel('Ratio (%)')
        axes[3].set_xlabel('Episode')

    # Agent action choice history (policy, innate)
    if hasattr(agent, 'probe'):
        action_history=np.array(agent.probe.sequence)
        map_reversed = U.reverse_dict(agent.probe.map) # dict
        for action_type in sorted(agent.probe.map.values()):
            axes[4].plot(np.cumsum(action_history==action_type), label=map_reversed[action_type])
        axes[4].legend()
        axes[4].set_title('Cumulative action types')
        axes[4].set_xlabel('Steps')
        axes[4].set_ylabel('Cumulative count')

        if save:
            filename = 'action_type.p'
            filename = filename if prefix=='' else f'{prefix}_{filename}'
            U.save_pickle(agent.probe, p/filename)

    # %%
    filename = 'result.png'
    filename = filename if prefix=='' else f'{prefix}_{filename}'
    if save: fig.savefig(p/filename)
    axes = axes.reshape(nrows, ncols)

    # Save result
    filename = 'result.xlsx'
    filename = filename if prefix=='' else f'{prefix}_{filename}'
    df.to_excel(p/filename)
    return fig, axes

# Not used.
def test_model(env, model, n_steps=1000):
    '''
    Test the model using specified number of steps.
    Use this for brief evaluations
    '''
    obs = env.reset()
    summary = {'reward':[], 'end':[]}
    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, end, info = env.step(action)
        summary['reward'].append(reward), summary['end'].append(end)
        if end:
            print('new episode')
            obs = env.reset()
    return summary
