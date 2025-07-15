from copy import deepcopy as dcopy
from typing import Optional
from types import SimpleNamespace

import gym
from gym import spaces
import numpy as np
from omegaconf import OmegaConf, DictConfig

try:
    import pygame
except:
    import warnings
    warnings.warn('pygame is not installed. The environment will break if render_mode="human"')

# %%
if False:
    pass
    # %%
    import os
    import hydra
    from omegaconf import OmegaConf, DictConfig

    PROJECT_DIR = os.getcwd()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'config'), job_name='debug')
    overrides = []
    cfg = hydra.compose(config_name='train_cfg', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    # Dummy class for debugging
    class Dummy():
        """ Dummy class for debugging """
        def __init__(self):
            pass
    self = Dummy()
    self.cfg = cfg

    # %%
    mapsize = cfg.env.mapsize
    hopsize = cfg.env.hopsize
    threshold_death = cfg.env.threshold_death
    threshold_reward = cfg.env.threshold_reward

    render_mode = 'human'
    self.metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}
    self.cfg = cfg.env

    # %%

# %%
class SimplePredatorEnvContinuous(gym.Env):
    """
    Predator, agent all moves equal distance.
    Goal does not move
    The predator is the part of the environment, not a separate agent.
    Map starts from (0,0), ends in (mapsize[0], mapsize[1]), and masize needs to be positive

    In case the agent reaches the goal and the predator at the same time, the agent is rewarded, not dead.

    (Abbreviations)
    P: Predator, G: Goal, A: Agent

    Functions starting with "_" are functions that are called within the class
    Functions without "_" are functions that may be called from the class object

    Parameters
    ----------
    mapsize: list of tuple of size (2,)
        Indicates the x, y limit of the map
    hopsize: float
        Indicates the maximum movement per each step
    threshold_death: float
        Distance threshold for the agent to get caught by the predator

    """
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 60}

    def __init__(self, mapsize=(10,10), hopsize_agent=1, hopsize_predator=1, threshold_death=1, threshold_reward=1, max_episode_steps=100, threshold_spawn=3, render_mode: Optional[str] = None, window_size=512):
        # Config setup
        self.cfg = DictConfig({'mapsize':mapsize, 'hopsize_agent':hopsize_agent, 'hopsize_predator':hopsize_predator, 'threshold_death':threshold_death, 'threshold_reward':threshold_reward, 'threshold_spawn':threshold_spawn}) # Static config that does not chagne over episodes (resets)
        self.cfg_run = DictConfig({'P_dir': (0,0)}) # Config for the run which stays static throughout the run, but changes every run (resets). May or may not be observed

        # TODO: Restrict action space to be circular, not square?
        self.action_space = spaces.Box(low=-self.cfg.hopsize_agent, high=self.cfg.hopsize_agent, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'A': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg.mapsize[0], self.cfg.mapsize[1]]), shape=(2,), dtype=np.float32),
            'P': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg.mapsize[0], self.cfg.mapsize[1]]), shape=(2,), dtype=np.float32),
            'G': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg.mapsize[0], self.cfg.mapsize[1]]), shape=(2,), dtype=np.float32),
        })
        self.reward_space = spaces.Discrete(3, start=-1)
        self.reward_range = (self.reward_space.start, self.reward_space.start + self.reward_space.n - 1)

        self.action_shape = 2
        self.observation_shape = 6
        self.reward_shape = None # Not array, but a single integer

        self.need_reset=True
        self.state = {'A': np.empty(2), 'G': np.empty(2), 'P': np.empty(2)} # internal states of the environment that varies each state. May or may not be observed

        # Render setup
        self.render = SimpleNamespace()
        self.render.render_mode = render_mode # Wanted to put this within self.render.cfg, but gym.Env has a namespace for self.render_mode.
        self.render.cfg = DictConfig({'window_size': window_size})

        if self.render.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.render.window = pygame.display.set_mode((self.render.cfg.window_size, self.render.cfg.window_size))
            self.render.clock = pygame.time.Clock()
        # self.render.renderer = Renderer(self.render_mode, self._render_frame)

        # Consistency check
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        assert np.all([n>0 for n in self.cfg.mapsize]), f'mapsize must be floats larger than 0, received: {self.cfg.mapsize}'

    def step(self, action):
        """
        Parameters
        ----------
        action: 2d float array of shape (2,) indicating the movement.
            Action will be clipped to maximum hopsize_agent

        Returns
        -------
        observation : dict of arrays
        reward: one of [-1,0,1]
        done: flag for the end of current episode
        info: additional info
        """
        assert not self.need_reset, 'Environment not initialized. Need to call env.reset() first.'

        # Adjusting action value
        action_distance = np.linalg.norm(action) # Just to avoid computing twice
        action = action if action_distance < self.cfg.hopsize_agent else action*(self.cfg.hopsize_agent/action_distance) # clip

        # Apply movements
        self.state['P'] += self.cfg_run.P_dir # Move
        self.state['A'] += action

        # Adjust environment setting
        # Clip position to be within mapsize
        for obj, pos in self.state.items():
            self.state[obj][0] = np.clip(pos[0], 0, self.cfg.mapsize[0]) # x
            self.state[obj][1] = np.clip(pos[1], 0, self.cfg.mapsize[1]) # y

        observation = self._get_obs()
        info = self._get_info()

        # If the P is on the edge, set new direction
        if not (0 < self.state['P'][0] < self.cfg.mapsize[0]) or not (0 < self.state['P'][1] < self.cfg.mapsize[1]):
            self.cfg_run.P_dir = self._reset_P()

        # Check reward condition, overrides done condition if they are both met
        if np.linalg.norm(self.state['A']-self.state['G']) < self.cfg.threshold_reward:
            reward=1
            terminated=True

        # Check terminated condition
        elif np.linalg.norm(self.state['A']-self.state['P']) < self.cfg.threshold_death:
            reward=-1
            terminated=True

        # Nothing happens
        else:
            reward=0
            terminated=False

        if self.render.render_mode=='human':
            self._render_frame()

        return observation, reward, terminated, info

    def _get_obs(self):
        '''
        In this case, the observation is the state.
        '''
        # return self.state.copy() # Return a copy in case to prevent modification of the state.
        return dcopy(self.state) # Return a copy in case to prevent modification of the state.

    def _get_info(self):
        return {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.need_reset=False

        # Set cfg_run
        self.cfg_run.P_dir = self._reset_P()

        # Set initial state
        def set_state():
            x = np.random.uniform(0,self.cfg.mapsize[0], size=len(self.state))
            y = np.random.uniform(0,self.cfg.mapsize[1], size=len(self.state))
            for i, obj in enumerate(self.state.keys()):
                self.state[obj][:] = x[i],y[i]

        set_state()
        # Invalid spawn points
        while np.linalg.norm(self.state['A']-self.state['G']) < self.cfg.threshold_spawn or np.linalg.norm(self.state['A']-self.state['P']) < self.cfg.threshold_spawn:
            set_state()

        observation = self._get_obs()

        if self.render.render_mode=='human':
            self._render_frame()

        return observation

    def _reset_P(self):
        angle = np.random.uniform(2*np.pi)
        return self.cfg.hopsize_predator*np.sin(angle).item(), self.cfg.hopsize_predator*np.cos(angle).item()

    def render(self):
        if self.render.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        # assert mode is not None  # The renderer will not call this function with no-rendering.
        # import pygame # avoid global pygame dependency. This method is not called with no-render.

        colors = {'P': 'r', 'G': 'g', 'A': 'b'} # Red, Green, Blue for Predator, Goal, Agent
        color_to_rgb = {'r': (255,0,0), 'g': (0,255,0), 'b': (0,0,255)}

        canvas = pygame.Surface((self.render.cfg.window_size, self.render.cfg.window_size))
        canvas.fill((255,255,255))

        # TODO: Convert to 2d
        pix_square_size = self.render.cfg.window_size / self.cfg.mapsize[0] # The number of pixels per unit mapsize

        for obj, pos in self.state.items():
            pygame.draw.circle(canvas, color_to_rgb[colors[obj]], pos*pix_square_size, pix_square_size/3)

        if self.render.render_mode == "human":
            assert self.render.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.render.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.render.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, 'window') and self.render.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

class SimplePredatorEnvContinuousInnateTraining(SimplePredatorEnvContinuous):
    """
    Predator, agent all moves equal distance.
    Goal does not move
    The predator is the part of the environment, not a separate agent.
    Map starts from (0,0), ends in (mapsize[0], mapsize[1]), and masize needs to be positive

    In case the agent reaches the goal and the predator at the same time, the agent is rewarded, not dead.

    (Abbreviations)
    P: Predator, G: Goal, A: Agent

    Functions starting with "_" are functions that are called within the class
    Functions without "_" are functions that may be called from the class object

    Parameters
    ----------
    mapsize: list of tuple of size (2,)
        Indicates the x, y limit of the map
    hopsize: float
        Indicates the maximum movement per each step
    threshold_death: float
        Distance threshold for the agent to get caught by the predator

    """

    def __init__(self, mapsize=(10,10), hopsize=1, threshold_death=1, threshold_reward=1, max_episode_steps=100, threshold_spawn=2, render_mode: Optional[str] = None, window_size=512):
        # Config setup
        self.cfg = DictConfig({'mapsize':mapsize, 'hopsize':hopsize, 'threshold_death':threshold_death, 'threshold_reward':threshold_reward, 'threshold_spawn':threshold_spawn, 'max_episode_steps':max_episode_steps}) # Static config that does not chagne over episodes (resets)
        self.cfg_run = DictConfig({'P_dir': (0,0)}) # Config for the run which stays static throughout the run, but changes every run (resets). May or may not be observed

        # TODO: Restrict action space to be circular, not square?
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'A': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg.mapsize[0], self.cfg.mapsize[1]]), shape=(2,), dtype=np.float32),
            'P': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg.mapsize[0], self.cfg.mapsize[1]]), shape=(2,), dtype=np.float32),
        })
        self.reward_space = spaces.Discrete(3, start=-1)
        self.reward_range = (self.reward_space.start, self.reward_space.start + self.reward_space.n - 1)

        self.action_shape = 2
        self.observation_shape = 4
        self.reward_shape = None # Not array, but a single integer

        self.need_reset=True
        self.state = {'A': np.empty(2), 'P': np.empty(2)} # internal states of the environment that varies each state. May or may not be observed

        # Render setup
        self.render = SimpleNamespace()
        self.render.render_mode = render_mode # Wanted to put this within self.render.cfg, but gym.Env has a namespace for self.render_mode.
        self.render.cfg = DictConfig({'window_size': window_size})

        if self.render.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.render.window = pygame.display.set_mode((self.render.cfg.window_size, self.render.cfg.window_size))
            self.render.clock = pygame.time.Clock()
        # self.render.renderer = Renderer(self.render_mode, self._render_frame)

        # Consistency check
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        assert np.all([n>0 for n in self.cfg.mapsize]), f'mapsize must be floats larger than 0, received: {self.cfg.mapsize}'

    def step(self, action):
        """
        Parameters
        ----------
        action: 2d float array of shape (2,) indicating the movement.
            Action will be clipped to maximum hopsize

        Returns
        -------
        observation : dict of arrays
        reward: one of [-1,0,1]
        done: flag for the end of current episode
        info: additional info
        """
        assert not self.need_reset, 'Environment not initialized. Need to call env.reset() first.'

        # Adjusting action value
        action_distance = np.linalg.norm(action) # Just to avoid computing twice
        action = action if action_distance < self.cfg.hopsize else action*(self.cfg.hopsize/action_distance) # clip

        # Apply movements
        self.state['P'] += self.cfg_run.P_dir # Move
        self.state['A'] += action

        # Adjust environment setting
        # Clip position to be within mapsize
        for obj, pos in self.state.items():
            self.state[obj][0] = np.clip(pos[0], 0, self.cfg.mapsize[0]) # x
            self.state[obj][1] = np.clip(pos[1], 0, self.cfg.mapsize[1]) # y

        observation = self._get_obs()
        info = self._get_info()

        # If the P is on the edge, set new direction
        if not (0 < self.state['P'][0] < self.cfg.mapsize[0]) or not (0 < self.state['P'][1] < self.cfg.mapsize[1]):
            self.cfg_run.P_dir = self._reset_P()

        # Check reward condition, overrides done condition if they are both met
        if np.linalg.norm(self.state['A']-self.state['P']) < self.cfg.threshold_death:
            reward=-1
            terminated=True

        # Nothing happens
        else:
            reward=0
            terminated=False

        if self.render.render_mode=='human':
            self._render_frame()

        return observation, reward, terminated, info

    def reset(self, seed=None):
        # super().reset(seed=seed)

        self.need_reset=False

        # Set cfg_run
        self.cfg_run.P_dir = self._reset_P()

        # Set initial state
        def set_state():
            x = np.random.uniform(0,self.cfg.mapsize[0], size=len(self.state))
            y = np.random.uniform(0,self.cfg.mapsize[1], size=len(self.state))
            for i, obj in enumerate(self.state.keys()):
                self.state[obj][:] = x[i],y[i]

        set_state()
        # Invalid spawn points
        while np.linalg.norm(self.state['A']-self.state['P']) < self.cfg.threshold_spawn:
            set_state()

        observation = self._get_obs()

        if self.render.render_mode=='human':
            self._render_frame()

        return observation
