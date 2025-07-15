# %%
from collections import deque
from copy import deepcopy as dcopy
import logging
import multiprocessing
from tqdm import tqdm
import functools

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import stable_baselines3
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# %%
import agents
import dict_operations as O
import environment
import eval as E
import trainer
import agent_models
import utils as U

# %%
log = logging.getLogger(__name__)

# %%
if False:
    pass
    # %%
    import os
    PROJECT_DIR = os.getcwd()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'config'), job_name='debug')

    # %%
    overrides = [
    '+exp/initial=common',
    'agent=Simple',
    '+agent/models/nn_actorcritic=transformer',
    'agent.actorcritic.features_dim=8',
    'train.update_interval=512',
    'agent.actorcritic.cfg_train.epoch=10',
    'agent.actorcritic.separate_input=False',
    ]
    # %%
    overrides = [
    '+exp/initial=common',
    'agent=Simple',
    '+agent/models/nn_actorcritic=transformer',
    'agent.actorcritic.features_dim=8',
    # 'train.update_interval=512',
    # 'agent.actorcritic.cfg_train.epoch=10',
    'agent.actorcritic.separate_input=False',
    ]
    # %%
    overrides = [
    '+exp/initial=common',
    'agent=Simple',
    # '+agent/models/nn_actorcritic=transformer',
    # 'agent.actorcritic.features_dim=8',
    # 'train.update_interval=512',
    # 'agent.actorcritic.cfg_train.epoch=10',
    # 'agent.actorcritic.separate_input=False',
    ]
    # %%
    overrides = [
    '+exp/initial/PredatorEnvContinuous=common',
    'agent=Innate',
    ]
    # %%
    cfg = hydra.compose(config_name= 'train_cfg', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    # %%
    # Configure logging: add stdout to log.info
    import sys
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.info('logging set')

    # Dummy class for debugging
    class Dummy():
        """ Dummy class for debugging """
        def __init__(self):
            pass
    self = Dummy()

# %%
def get_device(gpu_id=None):
    # return next(model.parameters()).device
    if gpu_id == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        p=multiprocessing.current_process()
        # Multirun
        try:
            worker_num = int(p.name.split('-')[-1]) # Fails if it's not integer
            gpu_id = worker_num % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        # Single run
        except ValueError: # Parent process
            if gpu_id == None:
                device = torch.cuda.default_stream().device
            else:
                device = torch.device(f'cuda:{gpu_id}')

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    return device

def setup(cfg):
    import socket, os
    # Node configurations
    log.info(f'[Node: {socket.gethostname()}][cwd: {os.getcwd()}]')

    # Print configs for debug purposes
    log.info(OmegaConf.to_yaml(cfg))
    # Set random seed
    U.seed(cfg.run.random_seed, strict=cfg.run.random_strict) # Set seed

    # Set paths for experiment
    path = U.Path()
    path.result = 'result'
    path.tensorboard = 'tensorboard'
    path.makedirs()
    tensorboard_writer = SummaryWriter(path.tensorboard)

    # GPU/CPU device
    device = get_device(cfg.gpu_id)
    log.info(device)

    return path, device, tensorboard_writer

def train(agent, env, cfg, episode=None, steps=None):
    """
    Run and train one episode using the given agent, env, and cfg.
    Note that agent.observe() is placed in a position where the data (observation, reward, end, info, action) are synchronized which ensures the following:

    o_t, r_t, d_t, i_t --[Agent]--> a_t --[Envrionment]--> o_t+1, r_t+1, d_t+1, i_t+1 ...

    Parameters
    ----------
    agent : an Agent instance.
    env: an Environment instance
    cfg: configurations

    Returns
    -------
    summary : dict with summary of the episodes (which may relate to metrics)
    """
    step_sum = 0
    agent.train() # Currently does nothing. Is this necessary?

    l_train_summary = []
    for episode in range(1, cfg.train.episode+1):
        # Preparation for the episode
        observation = env.reset()
        agent.reset_workingmemory()
        reward, end, info = None, False, None
        step = 0
        reward_list = [] # Reserved for summary

        while not end: # Timeout managed by TimeLimit Wrapper of env. Dealing wiht infinite loop is up to the env.
            # Interact with environment
            action = agent.act(observation, stochastic=True)
            agent.observe(observation=observation, reward=reward, end=end, info=info, action=action) # Store into memory
            observation, reward, end, info = env.step(action)

            # Training info & summary
            step_sum +=1
            step += 1
            reward_list.append(reward)

            # Update neural networks
            if step_sum % cfg.train.update_interval==0:
                log.info(f'training... {step_sum}')
                agent.update()

        agent.observe(observation=observation, reward=reward, end=True, info=info, action=None) # Agent needs to observe the end state
        reward = sum(reward_list)

        summary = {'episode': episode, 'step': step, 'reward': reward, 'end': end}
        log.info(summary)
        l_train_summary.append(summary)

    df_train_results = pd.DataFrame(U.merge_dict(l_train_summary))
    return df_train_results

def test(agent, env, cfg):
    """
    Run one episode using the given agent, env, and cfg.
    Note that agent.observe() is placed in a position where the data (observation, reward, end, info, action) are synchronized which ensures the following:

    o_t, r_t, d_t, i_t --[Agent]--> a_t --[Envrionment]--> o_t+1, r_t+1, d_t+1, i_t+1 ...

    Parameters
    ----------
    agent : an Agent instance.
    env: an Environment instance
    cfg: configurations

    Returns
    -------
    summary : dict with summary of the episodes (which may relate to metrics)
    """
    agent.eval() # Currently does nothing. Is this necessary?

    # Temporary for debug purposes
    l_observation_history = []

    l_test_summary = []
    for episode in tqdm(range(1, cfg.eval.episode+1)):
        # Preparation for the episode
        observation = env.reset()
        agent.reset_workingmemory()
        end = False
        step = 0
        reward_list = [] # Reserved for summary

        observation_history = [observation]

        while not end: # Timeout managed by TimeLimit Wrapper of env. Dealing wiht infinite loop is up to the env.
            # Interact with environment
            action = agent.act(observation)
            observation, reward, end, info = env.step(action)

            reward_list.append(reward)
            observation_history.append(observation)
            step += 1

        reward = sum(reward_list)

        if cfg.debug and reward==-1:
            l_observation_history.append(observation_history)

        summary = {'episode': episode, 'step': step, 'reward': reward, 'end': end}
        log.debug(summary)
        l_test_summary.append(summary)

    if cfg.debug: U.save_pickle(l_observation_history, 'observation_history.p')

    df_test_results = pd.DataFrame(U.merge_dict(l_test_summary))
    return df_test_results

def construct_nn_models(AgentClass, cfg, models, hack, info):
    '''
    Make nn objects & connect them to models
    - nn is the neural networks which acts like brain regions.
    - Note that neural networks should only receive & output scaled values, and unscaling should be performed outside of the neural network.
    '''

    nn_actorcritic = hack
    observation_space, action_space, reward_space = info['observation_space'], info['action_space'], info['reward_space']
    nn_models = {}

    for nn_name in AgentClass.nn_keys:
        if nn_name=='worldmodel':
            # ModelClass = eval(cfg.agent.worldmodel.ModelClass._target_)
            model = models['worldmodel']

            observation_sample, action_sample, reward_sample = observation_space.sample(), action_space.sample(), reward_space.sample()

            # dim_input
            observation_history = agents.construct_observation_history(observation=observation_sample, timewindow=model.timewindow, workingmemory=deque())
            observation_processed, action_processed = model.preprocess(observation_history=O.add_batch_dim(observation_history), action=O.add_batch_dim(action_sample)) # Preprocess sample observation
            dim_input = observation_processed.shape[1]+action_processed.shape[1] # Preprocessed dim_observation + action_dim

            # dim_observation
            observation_concatenated = np.concatenate(list(observation_sample.values()), axis=0) if isinstance(observation_sample, dict) else observation_sample # to get the observation dimension
            dim_observation = observation_concatenated.shape[0]

            # dim_reward
            if model.reward_type == 'category':
                dim_reward = reward_space.n
            else:
                raise Exception(f'Unknown reward_pred_type: {model.reward_pred_type}')

            log.debug(f'[WorldModel] dim_input: {dim_input}, output_dim: {dim_observation+dim_reward}')

            nn_model = hydra.utils.instantiate(cfg.agent.worldmodel.NNClass, dim_observation=dim_observation, dim_reward=dim_reward)
            nn_models[nn_name] = nn_model
            model.nn = nn_model
        elif nn_name=='actorcritic':
            nn_models[nn_name] = nn_actorcritic
        elif nn_name=='selfmodel':
            # ModelClass = eval(cfg.agent.selfmodel.ModelClass._target_)
            model = models['selfmodel']

            # observation_sample, action_sample = O.add_batch_dim(observation_space.sample()), O.add_batch_dim(action_space.sample())
            observation_sample, action_sample = observation_space.sample(), action_space.sample()

            observation_history = agents.construct_observation_history(observation=observation_sample, timewindow=model.timewindow, workingmemory=deque())
            observation_processed = model.preprocess(observation_history=O.add_batch_dim(observation_history))

            dim_observation = observation_processed.shape[1]
            dim_action = action_sample.shape[0]
            dim_value = 1 # Only single value function for now.

            nn_model = hydra.utils.instantiate(cfg.agent.selfmodel.NNClass, dim_observation=dim_observation, dim_action=dim_action, dim_value=dim_value)
            nn_models[nn_name] = nn_model
            model.nn = nn_model
        else:
            raise ValueError(f'Unknown nn_name of Agent: {nn_name}')
    nn_models = nn.ModuleDict(nn_models)
    return nn_models

def construct_models(AgentClass, cfg, hack, info):
    '''
    Make models
    - model is the functional unit to be used in inference
    - model takes care of preprocess, inference through the nn, and postprocess
    - models need nn, which will be connected in construct_nn_models()
    '''
    model_actorcritic = hack
    observation_space, action_space, reward_space = info['observation_space'], info['action_space'], info['reward_space']
    models = {}

    for model_name in AgentClass.model_keys:
        if model_name=='worldmodel':
            # ModelClass = eval(cfg.agent.worldmodel.ModelClass._target_)
            # models[model_name] = ModelClass(nn=None, timewindow=cfg.agent.worldmodel.timewindow)
            models[model_name] = hydra.utils.instantiate(cfg.agent.worldmodel.ModelClass, nn=None, amp=cfg.run.amp, timewindow=cfg.agent.worldmodel.timewindow)

        elif model_name=='value':
            models[model_name] = model_actorcritic['value']
        elif model_name=='policy':
            models[model_name] = model_actorcritic['policy']
        elif model_name=='selfmodel':
            # ModelClass = eval(cfg.agent.selfmodel.ModelClass._target_)
            # models[model_name] = ModelClass(nn=None)

            models[model_name] = hydra.utils.instantiate(cfg.agent.selfmodel.ModelClass, nn=None, action_space=action_space, amp=cfg.run.amp, timewindow=cfg.agent.selfmodel.timewindow)
        else:
            raise ValueError(f'Unknown model_name of Agent: {model_name}')
    return models

def construct_trainers(AgentClass, cfg, models, nn_models, hack, tensorboard_writer):
    '''
    Make trainers
    - trainer trains the nn using gradient descent.
    '''
    trainer_actorcritic = hack
    trainers = {}
    for nn_name in AgentClass.nn_keys:
        if nn_name=='worldmodel':
            # ModelClass = eval(cfg.agent.worldmodel.ModelClass._target_)
            TrainerClass = eval(cfg.agent.worldmodel.TrainerClass._target_)
            DatasetClass = eval(cfg.agent.worldmodel.TrainerClass.DatasetClass._target_) # Class
            CriterionClass = eval(cfg.agent.worldmodel.TrainerClass.CriterionClass._target_)
            kwargs_trainer = {'nn': nn_models[nn_name], 'model': models['worldmodel'], 'cfg_train':cfg.agent.worldmodel.TrainerClass.cfg_train, \
                'DatasetClass':DatasetClass, 'CriterionClass':CriterionClass, 'name':nn_name, 'tensorboard_writer': tensorboard_writer, 'amp': cfg.run.amp}
            trainers[nn_name] = TrainerClass(**kwargs_trainer)
        elif nn_name=='actorcritic':
            trainers[nn_name] = trainer_actorcritic
        elif nn_name=='selfmodel':
            # ModelClass = eval(cfg.agent.worldmodel.ModelClass._target_)
            TrainerClass = eval(cfg.agent.selfmodel.TrainerClass._target_)
            DatasetClass = eval(cfg.agent.selfmodel.TrainerClass.DatasetClass._target_) # Class
            CriterionClass = eval(cfg.agent.selfmodel.TrainerClass.CriterionClass._target_)
            kwargs_CriterionClass = {k: v for k, v in OmegaConf.to_container(cfg.agent.selfmodel.TrainerClass.CriterionClass).items() if k != '_target_'}
            CriterionClass = functools.partial(CriterionClass, **kwargs_CriterionClass)
            kwargs_trainer = {'nn': nn_models[nn_name], 'model': models['selfmodel'], 'cfg_train':cfg.agent.selfmodel.TrainerClass.cfg_train, \
                'DatasetClass':DatasetClass, 'CriterionClass':CriterionClass, 'name':nn_name, 'tensorboard_writer': tensorboard_writer, 'amp': cfg.run.amp}
            trainers[nn_name] = TrainerClass(**kwargs_trainer)
        else:
            raise ValueError(f'Unknown nn_name of Agent: {nn_name}')
    return trainers

# %%
@hydra.main(config_path='config', config_name='train_cfg', version_base='1.1')
def main(cfg: DictConfig) -> None: # Load configs automatically
    pass

    # %%
    path, device, tensorboard_writer = setup(cfg) # print cfg, setup random seed, make directories
    log.info(f'[Device: {device}][tensorboard_writer: {tensorboard_writer}]')

    # %%
    env = hydra.utils.instantiate(cfg.env) # No rendering during training, cuz we need lots of experience
    info = {'observation_space': env.observation_space, 'action_space': env.action_space, 'reward_space': env.reward_space}

    # %%
    # It is possible to code cfg.agent.trainers so that hydra.utils.instantiate(cfg.agent.trainers.${name}) would take care of everything. But for readability, explicitly specify the module classes involved.
    import stable_baselines3 # joblib multiprocess bug, Somehow it can't find the modules imported from the parent process.
    import agents
    TrainerType = eval(cfg.agent.actorcritic.TrainerType._target_)
    AgentClass = eval(cfg.AgentClass._target_)

    # %%
    # Hack stable baselines
    nn_actorcritic, model_actorcritic, trainer_actorcritic = agent_models.hack_stablebaselines(TrainerType=TrainerType, env=env, cfg=cfg, tensorboard_writer=tensorboard_writer)
    log.info(f'TrainerType: {TrainerType}') # == print() (https://hydra.cc/docs/tutorials/basic/running_your_app/logging/)

    # %%
    models = construct_models(AgentClass=AgentClass, cfg=cfg, hack=model_actorcritic, info=info) # Should all construct_ functions include "info" argument for consistency?
    log.info(f'models: {models}')

    nn_models = construct_nn_models(AgentClass=AgentClass, cfg=cfg, models=models, hack=nn_actorcritic, info=info)
    log.info(f'nn_models: {nn_models}')

    trainers = construct_trainers(AgentClass=AgentClass, cfg=cfg, models=models, nn_models=nn_models, hack=trainer_actorcritic, tensorboard_writer=tensorboard_writer)
    log.info(f'trainers: {trainers}')

    # %%
    OmegaConf.resolve(cfg) # Substitute variable interpolatiogn
    agent = AgentClass(cfg=cfg.agent, nn=nn_models, trainers=trainers, models=models, observation_preprocessor=None, info=info)
    log.info(f'[Agent]\n{agent}')

    # %%
    # Training episodes
    agent.nn.to(device)
    df_train_results = train(agent, env, cfg)
    log.info(f'# of training steps: {df_train_results["step"].sum()}')

    # %%
    # Evaluate training episodes
    E.evaluate(df=df_train_results, cfg=cfg, agent=agent, save_dir=path.result, prefix='train')

    # %%
    # Testing episodes
    # TODO: incorporate render arguments within cfg, such as render_mode, window_size
    render_mode = 'human' if cfg.eval.visualize else None
    env = hydra.utils.instantiate(cfg.env, env={'render_mode': render_mode})
    env.metadata['render_fps']=40

    agent.reset_probe()
    df_test_results = test(agent, env, cfg)

    # Evaluate testing episodes
    E.evaluate(df=df_test_results, cfg=cfg, agent=agent, save_dir=path.result, prefix='test')

# %%
if __name__=='__main__':
    main()
