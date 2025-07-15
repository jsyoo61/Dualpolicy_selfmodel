from collections import deque
from copy import deepcopy as dcopy
import logging

import gym
import gym.spaces as spaces
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

log = logging.getLogger(__name__)

# %%
import dict_operations as O
import trainer
import utils as U

# %%
# TODO: Maybe Just put all worldmodel modules into worldmodel.py?
# TODO (Important): Modify trainer so that it will receive arguments istead of "data"
def weighted_crossentropy_loss(y, classes):
    y_type, y_count = torch.unique(y, return_counts=True)
    assert set(classes).issuperset(set(y_type.numpy())), f'invalid y_type found: [{y_type}/{classes}]'

    # Classes need to be specified since
    weight = torch.zeros(len(classes))
    for y_type_, y_count_ in zip(y_type, y_count):
        i = classes.index(y_type_.item())
        weight[i] = 1/y_count_

    weight /= weight.sum()
    return nn.CrossEntropyLoss(weight=weight)

# %%
def nan_idx_converter(d):
    """
    return a idx map (hash table in form of a list) which redirects the idx to the next non-nan value

    Parameters
    ----------
    d : np.ndarray or torch.Tensor
    _convert_tensor: whether d is a tensor or not

    Returns
    -------
    idx_new : np.ndarry which indicates the next idx with d[idx]!=float('nan')
    """
    return np.where(~np.isnan(d))[0]

def get_action_distribution(mean_action, std_action, action_space):
    """
    (Explanation here)

    Parameters
    ----------
    arg : array-like of shape (n_samples,), default=None
        Argument explanation.
        If ``None`` is given, those that appear at least once
        .. versionadded:: 0.18

    Returns
    -------
    action_distribution : torch.distributions.Distribution object
    """
    if isinstance(action_space, gym.spaces.Box):
        return torch.distributions.Normal(mean_action, std_action, validate_args=False)
    else:
        raise ValueError('Invalid action distribution')

# %%
class FNN(nn.Module):
    '''
    basic FNN module

    Parameters
    ----------
    n_input: number of input (int)
        if n_input is None, then the LazyLinear() is set in the first layer.
    n_hidden_list: list of hidden neurons (list of int)
    activation_list: torch.nn activation function instances (nn object or list of nn object)

    Example
    -------
    >>> model = FNN(n_input=10, n_hidden_list=[8,6,5], activation_list=[nn.Sigmoid(), nn.ReLU(), nn.Tanh()])
    # n_hidden_list, activation_list corresponds to [h1, h2, output]
    >>> print(model)
    FNN(
      (fc): Sequential(
        (0): Linear(in_features=10, out_features=8, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=8, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=5, bias=True)
        (5): Tanh()
      )
    )
    '''
    def __init__(self, n_hidden_list, activation_list, n_input=None):
        super().__init__()
        if type(activation_list) is not list:
            activation_list = [dcopy(activation_list)]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'

        # 1st layer - Select Lazy if n_input is not specified
        if n_input is None:
            layers = [nn.Flatten(1), nn.LazyLinear(n_hidden_list[0]), activation_list[0]]
        else:
            layers = [nn.Flatten(1), nn.Linear(n_input, n_hidden_list[0]), activation_list[0]]
        # Hidden layers ~ Output layer
        for i in range(len(n_hidden_list) - 1):
            layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]), activation_list[i+1]])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        '''x.shape()==(batch_size, feature_dim)'''
        return self.fc(x)

# %%
class BaseModel(object):
    call_arguments = []

    def __init__(self, nn, amp=False):
        self.nn = nn
        self.amp = amp

    def __call__(self, batch=False, *args, **kwargs):
        """
        The preprocess/postprocess assumes batch dimension always exists.

        Parameters
        ----------
        batch : bool, default=False
            Whether the input has the batch dimension.
            If False, a batch dimension (axis=0) will be added
        *args, **kwargs : argument specific to each model

        Returns
        -------
        arrays : tuple of ndarrays
            The return value of a model
        """
        # Trim call_arguments (which are mostly not np.ndarray)
        call_arguments = {k:v for k, v in kwargs.items() if k in self.call_arguments}
        kwargs = {k:v for k, v in kwargs.items() if k not in self.call_arguments}

        # Setup inference
        self.nn.eval()
        device = U.get_device(self.nn)
        if not batch: # If the given values are does not have batch dimension,
            args = tuple(O.add_batch_dim(arg) for arg in args)
            kwargs = {k: O.add_batch_dim(v) for k, v in kwargs.items()}

        # Preprocess & Convert to torch.Tensor
        arrays = self.preprocess(*args, **kwargs)
        arrays = arrays if isinstance(arrays, tuple) else (arrays,)
        tensors = [torch.as_tensor(array, device=device) for array in arrays]

        # Forward pass without gradient & Convert to numpy.ndarray
        if self.amp:
            with torch.autocast(device_type=device.type):
                with torch.no_grad(): tensors = self.forward(*tensors, **call_arguments)
        else:
            with torch.no_grad(): tensors = self.forward(*tensors, **call_arguments)
        tensors = tensors if isinstance(tensors, tuple) else (tensors,)
        arrays = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]

        # Postprocess
        arrays = self.postprocess(*arrays)
        if not batch:
            arrays = tuple(O.remove_batch_dim(array) if isinstance(array, (np.ndarray, dict)) else array for array in arrays) if isinstance(arrays, tuple) else O.remove_batch_dim(arrays)

        return arrays # May be tuple or numpy.ndarray

    def forward(self, *args, **kwargs):
        '''
        May be overridden in children classes
        '''
        return self.nn(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        '''
        Preprocess the arguments into a form that NN could compute.
        '''
        return *args, *kwargs.values()

    def inv_preprocess(self, *args, **kwargs):
        return *args, *kwargs.values()

    def postprocess(self, *args, **kwargs):
        '''
        Preprocess the arguments into
        '''
        return *args, *kwargs.values()

    def inv_postprocess(self, *args, **kwargs):
        return *args, *kwargs.values()

# %%
# World Model
class WorldModel(BaseModel):
    def __init__(self, nn, amp=False, timewindow=2):
        super().__init__(nn=nn, amp=amp)
        self.timewindow = timewindow
        self.reward_type = 'category'
        self.reward_pred_type = 'hard'
        assert self.timewindow>=1, f'timewindow should be >=1, received: {self.timewindow}'

    def preprocess(self, observation_history, action):
        """

        Parameters
        ----------
        observation_history : dict of numpy.ndarrays with shape (Batch, time, observation_space)
        action : numpy.ndarray of shape (Batch, action_space)

        Returns
        -------
        observation_processed : ndarray of shape (Batch, features)
            observation with concatenated coordinates [:,0:6], and distances [:,6:9]
        action_processed: identical action
        """
        assert all([observation.ndim==3 for observation in observation_history.values()])
        assert U.equal([observation.shape[1] for observation in observation_history.values()]), 'time dimension in all observation should be identical'
        assert observation_history['A'].shape[1]==self.timewindow, f'observation_history must have same timepoints ({observation_history["A"].shape[1]}) as self.timewindow ({self.timewindow})'

        # Previous observation coordinates
        observation_history_concatenated = concat_observation(observation_history) # Dict of shape (Batch, time, 6)
        observation_history_coordinates = observation_history_concatenated.reshape(len(observation_history_concatenated),-1) # shape (Batch, 6*time)

        # Final observation distances
        observation_last = {key: observation[:,-1,:] for key, observation in observation_history.items()} # dict of shape (Batch, 2)
        observation_last_distance = observation_to_distance(observation_last) # shape (Batch, 3)

        observation_history_processed = np.concatenate((observation_history_coordinates, observation_last_distance), axis=1, dtype=np.float32) # (Batch, 6*time + 3)

        return observation_history_processed, action

    def inv_preprocess(self, observation_history, action):
        observation_coordinates = observation_history[:,-9:-3]
        # observation_coordinates, observation_distance = observation_processed[:,:6], observation_processed[:,6:]
        observation_coordinates = {'A': observation_coordinates[:,:2], 'G': observation_coordinates[:,2:4], 'P': observation_coordinates[:,4:]}
        return observation_coordinates, action

    def postprocess(self, observation_next_hat, reward_next_hat):
        """
        Parameters
        ----------
        observation_next_hat : array of shape (Batch, 6)
            Output of WorldModelNet.
        reward_next_hat : array of shape (Batch, 3)
            Output of WorldModelNet

        Returns
        -------
        C : ndarray of shape (n_classes, n_classes)
            Explanation variable
        """
        # observation_next_hat
        observation_next_hat_processed = {'A': observation_next_hat[:,:2], 'G': observation_next_hat[:,2:4], 'P': observation_next_hat[:,4:]}

        # reward
        if self.reward_pred_type=='soft':
            reward_next_hat_processed = -reward_next_hat[:,0] + reward_next_hat[:,2]
        elif self.reward_pred_type=='hard':
            reward_next_hat_processed = np.argmax(reward_next_hat) - 1 # Convert [0, 1, 2] into [-1, 0, 1]

        return observation_next_hat_processed, reward_next_hat_processed

    def inv_postprocess(self, observation, reward):
        observation_coordinates = concat_observation(observation)
        return observation_coordinates, reward

class WorldModelNet(nn.Module):
    def __init__(self, dim_observation, dim_reward, n_hidden_list=[32]): # TODO: Specify arguments for self.layer so that we could do architecture search
        '''
        observation_sample: numpy array of ndim==1 (preprocessed observation from SimplePredatorEnvObsPreprocessor)
        '''
        super().__init__()
        # assert observation_sample.ndim==2, f'Sample observation should have dimensions: (Batch, dim_observation), received: {observation_sample.shape}'
        # dim_observation = observation_sample.shape[1:]

        # 1st layer is Lazylinear so no need to specify input shape, output shape is dim_observation+reward_dim
        self.layer = FNN(n_hidden_list=n_hidden_list+[dim_observation+dim_reward], activation_list=[nn.ReLU()]*len(n_hidden_list)+[nn.Identity()])
        self.dim_observation = dim_observation
        self.dim_reward = dim_reward

    def forward(self, observation_history_processed, action):
        """
        World model forward function should act like env.step()

        Parameters
        ----------
        observation_history_processed: tensor (Batch, observation_history_feature_shape)
            observation_history_feature_shape is determined by WorldModelModel.preprocess()
        action: tensor (Batch, action_shape)

        Returns
        -------
        observation_next_hat: tensor of shape (Batch, observation_shape)
        reward: tensor of shape (Batch, 1)
        """
        x = torch.cat((observation_history_processed, action), dim=-1)
        y_hat = self.layer(x)
        observation_next_hat, reward_next_hat_prob = y_hat[:,:self.dim_observation], y_hat[:,-self.dim_reward:]

        return observation_next_hat, reward_next_hat_prob

class WorldModelDataset(D.Dataset):
    # classes = [-1, 0, 1]
    classes = [0, 1, 2]
    def __init__(self, memory, model, convert_tensor=True):
        """
        Training a World model needs
        - input: observation, action
        - target: observation_next, reward_next
        *Be careful of data dtypes

        Returned dictionary of data is passed to Trainer().forward(data)

        Parameters
        ----------
        observation : array-like of shape (n_samples, dim_observation)
        action : array-like of shape (n_samples, dim_action)
        reward : array-like of shape (n_samples,)
        convert_tensor: bool, default=True
            Saves the received arrays into torch tensors.
        """
        observation, action, reward, end = memory['observation'], memory['action'], memory['reward'], memory['end']

        l_observation_history_processed, l_action, l_observation_next, l_reward_next = [], [], [], []
        for observation_history, action_, observation_next, reward_next, end_history in \
        zip(O.nwise(O.__getitem__(observation, slice(None,-1)), n=model.timewindow), action[model.timewindow-1:-1], O.__iter__(O.__getitem__(observation, slice(model.timewindow,None))), reward[model.timewindow:], O.nwise(end[:-1], n=model.timewindow)):
            if not end_history.any(): # Check proper transition, if all previous observations were not terminal observations:
                observation_history_processed, action_processed = model.preprocess(observation_history=O.add_batch_dim(observation_history), action=O.add_batch_dim(action_)) # Preprocess sample observation
                observation_history_processed, action_processed = observation_history_processed.squeeze(0), action_processed.squeeze(0)
                observation_next_processed, reward_next_processed = model.inv_postprocess(observation=observation_next, reward=reward_next)

                l_observation_history_processed.append(observation_history_processed), l_action.append(action_processed), l_observation_next.append(observation_next_processed), l_reward_next.append(reward_next_processed)

        observation_history_processed, action, observation_next, reward_next = np.array(l_observation_history_processed), np.array(l_action), np.array(l_observation_next), np.array(l_reward_next)
        reward_next = (reward_next+1).astype(np.int64) # convert [-1, 0, 1] into categories of [0, 1, 2]

        self.observation_history_processed = torch.as_tensor(observation_history_processed)
        self.action = torch.as_tensor(action)
        self.observation_next = torch.as_tensor(observation_next)
        self.reward_next = torch.as_tensor(reward_next)

    def __getitem__(self, idx):
        observation_history_processed = self.observation_history_processed[idx]
        action = self.action[idx]
        observation_next = self.observation_next[idx]
        reward_next = self.reward_next[idx]
        return {'observation_history_processed': observation_history_processed, 'action': action, 'observation_next': observation_next, 'reward_next': reward_next}

    def __len__(self):
        return len(self.reward_next)

class WorldModelTrainer(trainer.Trainer):
    def forward(self, observation_history_processed, action, observation_next, reward_next):
        """
        The forward pass when training the World model.

        Parameters
        ----------
        data : dict of torch.Tensor
            data must include the following keys: [obervation, action, observation_next, reward_next]

        Returns
        -------
        result : dict of torch.Tensor
            tensors that are passed to Criterion, which are needed to compute the loss value.
        """
        observation_next_hat, reward_next_hat_prob = self.nn(observation_history_processed=observation_history_processed, action=action)
        loss_observation, loss_reward = self.criterion(observation_next=observation_next, observation_next_hat=observation_next_hat, reward_next=reward_next, reward_next_hat_prob=reward_next_hat_prob)

        log.debug(f'[WorldModel] loss_o: {loss_observation:.6f}, loss_r: {loss_reward:.6f}')
        if self.tensorboard_writer is not None:
            # Temporary, for debugging
            with torch.no_grad():
                loss_observation_A = F.mse_loss(observation_next[:,:2], observation_next_hat[:,:2]) #
                loss_observation_G = F.mse_loss(observation_next[:,2:4], observation_next_hat[:,2:4])
                loss_observation_P = F.mse_loss(observation_next[:,4:], observation_next_hat[:,4:])
                self.tensorboard_writer.add_scalar('worldmodel/loss_observation_A', loss_observation_A.item(), self.iter)
                self.tensorboard_writer.add_scalar('worldmodel/loss_observation_G', loss_observation_G.item(), self.iter)
                self.tensorboard_writer.add_scalar('worldmodel/loss_observation_P', loss_observation_P.item(), self.iter)

            self.tensorboard_writer.add_scalar('worldmodel/loss_observation', loss_observation.item(), self.iter)
            self.tensorboard_writer.add_scalar('worldmodel/loss_reward', loss_reward.item(), self.iter)

        return loss_observation+loss_reward

class WorldModelCriterion(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # TODO: Make the loss functions capable of changing
        self.crit_observation = nn.MSELoss()
        self.crit_reward = weighted_crossentropy_loss(y=dataset.reward_next, classes=dataset.classes) # Weight according to reward_next

        # TODO: Make lambda coefficient for weights of criterions
        # self.lambda

    def forward(self, observation_next, observation_next_hat, reward_next, reward_next_hat_prob):
        """
        Computes 2 types of criterion for the World model (observation, reward prediction).

        Parameters
        ----------
        observation_next : tensor of shape (n_samples, dim_observation)
            Ground truth next observation
        observation_next_hat: tensor of shape (n_samples, dim_observation)
            Predicted next observation
        reward_next: tensor of shape (n_samples,)
            Ground truth next reward
        reward_next_hat_prob: tensor of shape (n_samples,)
            Predicted next reward

        Returns
        -------
        loss : tensor of shape (1,)
        """
        loss_observation = self.crit_observation(observation_next_hat, observation_next)
        loss_reward = self.crit_reward(reward_next_hat_prob, reward_next)

        return loss_observation, loss_reward

def concat_observation(observation):
    """
    Preprocesses observation from SimplePredatorEnv so that the models of Agent could interpret the input

    Parameters
    ----------
    observation : dict with keywords [A, G, P], values of ndarray with ndim==2
        Observation obtained from SimplePredatorEnv().step()

    Returns
    -------
    observation_processed : single ndarray
        Preprocessed observation
    """
    keys = sorted(observation.keys())
    observation_processed = np.concatenate([observation[k] for k in keys], axis=-1, dtype=np.float32) # shape: (buffer_size, observation_dim)
    return observation_processed

def observation_to_distance(observation):
    """
    Preprocess observation into distances
    o1: A-P
    o2: P-T
    o3: T-A

    Parameters
    ----------
    observation : ndarray of shape (n_samples, dim_observation)

    Returns
    -------
    observation_easy : ndarray of shape (n_samples, 3)
        Observation converted into distances
    """
    # Compute the distance between three objects
    o1, o2, o3 = np.linalg.norm(observation['A'] - observation['P'], axis=1), np.linalg.norm(observation['P'] - observation['G'], axis=1), np.linalg.norm(observation['G'] - observation['A'], axis=1)
    # o1, o2, o3 = np.linalg.norm(observation[:,:2] - observation[:,2:4], axis=1), np.linalg.norm(observation[:,2:4] - observation[:,4:], axis=1), np.linalg.norm(observation[:,4:] - observation[:,:2], axis=1)
    observation_distance = np.stack((o1,o2,o3), axis=1).astype(np.float32)
    return observation_distance

# %%
# Self Model
# Distillation loss
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, T=2, reduction='mean'):
        '''
        :param weight: class weights
        '''
        super().__init__()
        self.T=T
        self.reduction=reduction

    def forward(self, y_hat, y_hat_true):
        '''
        :param y_hat: estimated value of model, shape: (N_batch, n_class)
        :param y_hat_true: soft target, shape: (N_batch, n_class)
        '''
        y_log_score = F.log_softmax(y_hat/self.T, dim=1)
        y_score_true = F.softmax(y_hat_true/self.T, dim=1)
        loss = -y_score_true * y_log_score
        if self.reduction=='mean':
            loss = loss.mean()
        elif self.reduction=='none':
            pass
        loss = loss * (self.T**2)
        return loss

class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, T=2, reduction='mean'):
        super().__init__()
        # TODO: Make weight parameter
        self.T=T
        self.reduction=reduction

    def forward(self, y_hat, y_hat_true):
        '''
        :param y_hat: 2D tensor of shape (N_batch, 1)
        :param y_hat_true: 2D tensor of shape (N_batch, 1)
        '''
        y_log_score_1 = F.logsigmoid(y_hat/self.T) # use logsigmoid() rather than log(sigmoid()) for floating point precision
        y_log_score_0 = F.logsigmoid(-y_hat/self.T) # log(1 - (1/(1+exp(-x)))) == log(1/(1+exp(x)))
        y_score_true = torch.sigmoid(y_hat_true/self.T)
        loss = -( y_score_true * y_log_score_1 + (1-y_score_true) * y_log_score_0 )
        if self.reduction=='mean':
            loss = loss.mean()
        elif self.reduction=='none':
            pass
        loss = loss * (self.T**2)
        return loss

class SelfModel(BaseModel):
    call_arguments = ['stochastic', 'return_distrib']
    def __init__(self, nn, action_space, amp=False, timewindow=2):
        super().__init__(nn=nn, amp=amp)
        self.action_space=action_space
        self.timewindow=timewindow

    def forward(self, observation_history, stochastic=False, return_distrib=False):

        action_hat, value_hat = self.nn(observation_history)

        if return_distrib:
            action_distribution = get_action_distribution(action_hat, self.nn.log_std_action.exp().expand_as(action_hat), self.action_space)
            action_hat = action_distribution
        elif stochastic:
            action_distribution = get_action_distribution(action_hat, self.nn.log_std_action.exp().expand_as(action_hat), self.action_space)
            action_hat = action_distribution.sample()
        else:
            action_hat = action_hat

        return action_hat, value_hat

    def preprocess(self, observation_history):
        """

        Parameters
        ----------
        observation : dict of numpy.ndarrays with shape (Batch, observation_space)

        Returns
        -------
        observation_processed : ndarray of shape (Batch, 6)
        """
        # Observation coordinates
        observation_history_concatenated = concat_observation(observation_history) # Dict of shape (Batch, time, 6)
        observation_history_coordinates = observation_history_concatenated.reshape(len(observation_history_concatenated),-1) # shape (Batch, 6*time)
        return observation_history_coordinates

    def inv_preprocess(self, observation):
        pass

    def postprocess(self, action, value):
        """
        Parameters
        ----------
        action : array of shape (Batch, 2)
            Output of SelfModelNet.
        value : array of shape (Batch, 1)
            Output of SelfModelNet

        Returns
        -------
        action : array of shape (Batch, 2)
            Scaled action
        value : array of shape (Batch, 1)
            Identical to input
        """
        if isinstance(self.action_space, gym.spaces.Box):
            action_scaled = np.clip(action, self.action_space.low, self.action_space.high)

        return action_scaled, value

    def inv_postprocess(self, action, value):
        # Do nothing
        return action, value

class SelfModelNet(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_value=1, n_hidden_list=[64,64]): # TODO: Specify arguments for self.layer so that we could do architecture search
        super().__init__()
        # 1st layer is Lazylinear so no need to specify input shape, output shape is dim_action+dim_value(1)
        self.layer = FNN(n_hidden_list=n_hidden_list+[dim_action+dim_value], activation_list=[nn.ReLU()]*len(n_hidden_list)+[nn.Identity()])
        self.log_std_action = nn.Parameter(torch.zeros(dim_action)) # requires_grad=True by default
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_value = dim_value

    def forward(self, observation_history):
        """
        Self model forward function should act like actorcritic()

        Parameters
        ----------
        observation_history: tensor (Batch, observation_dim)
            observation_dim determined by SelfModel.preprocess()

        Returns
        -------
        action: tensor (Batch, action_shape)
        value: tensor (Batch, 1)
        """
        y_hat = self.layer(observation_history)
        action_hat, value_hat = y_hat[:,:self.dim_action], y_hat[:,-self.dim_value:]

        return action_hat, value_hat

class SelfModelNetDouble(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_value=1, n_hidden_list=[64,64]): # TODO: Specify arguments for self.layer so that we could do architecture search
        super().__init__()
        # 1st layer is Lazylinear so no need to specify input shape, output shape is dim_action+dim_value(1)
        self.layer_action = FNN(n_hidden_list=n_hidden_list+[dim_action], activation_list=[nn.ReLU()]*len(n_hidden_list)+[nn.Identity()])
        self.layer_value = FNN(n_hidden_list=n_hidden_list+[dim_value], activation_list=[nn.ReLU()]*len(n_hidden_list)+[nn.Identity()])
        self.log_std_action = nn.Parameter(torch.zeros(dim_action)) # requires_grad=True by default
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_value = dim_value

    def forward(self, observation_history):
        """
        Self model forward function should act like actorcritic()

        Parameters
        ----------
        observation_history: tensor (Batch, observation_dim)
            observation_dim determined by SelfModel.preprocess()

        Returns
        -------
        action: tensor (Batch, action_shape)
        value: tensor (Batch, 1)
        """
        action_hat = self.layer_action(observation_history)
        value_hat = self.layer_value(observation_history)

        return action_hat, value_hat

class SelfModelDataset(D.Dataset):
    def __init__(self, memory, model, convert_tensor=True):
        """
        Training a Self model needs
        - input: observation
        - target: action, value
        *Be careful of data dtypes

        Returned dictionary of data is passed to Trainer().forward(data)

        Parameters
        ----------
        memory:
        model: SelfModel() instance
        """
        observation_history, action, value, end = memory['observation_history'], memory['action'], memory['value'], memory['end']

        l_observation_history, l_action, l_value = [], [], []
        for observation_history_, action_, value_, end_ in zip(O.__iter__(observation_history), action, value, end):
            if not end_: # Terminal step does not have an action
                observation_history_processed = O.remove_batch_dim(model.preprocess(observation_history=O.add_batch_dim(observation_history_)))
                action_processed, value_processed = model.inv_postprocess(action=action_, value=value_) # Does nothing but for consistency.

                l_observation_history.append(observation_history_processed), l_action.append(action_processed), l_value.append(value_processed)

        observation_history, action, value = np.array(l_observation_history), np.array(l_action), np.array(l_value)

        self.observation_history = torch.as_tensor(observation_history)
        self.action = torch.as_tensor(action)
        self.value = torch.as_tensor(value)[...,None]

    def __getitem__(self, idx):
        observation_history = self.observation_history[idx]
        action = self.action[idx]
        value = self.value[idx]
        return {'observation_history': observation_history, 'action': action, 'value': value}

    def __len__(self):
        return len(self.value)

class SelfModelTrainer(trainer.Trainer):
    def forward(self, observation_history, action, value):
        """
        The forward pass when training the World model.

        Parameters
        ----------
        observation : torch.Tensor of shape (Batch, dim_observation)
        action:

        value:

        Returns
        -------
        loss : single tensor
            A tensor to be called .backward()
        """
        action_hat, value_hat = self.nn(observation_history=observation_history)
        action_distribution = get_action_distribution(action_hat, self.nn.log_std_action.exp().expand_as(action_hat), self.model.action_space)

        loss_action, loss_value = self.criterion(action=action, action_distribution=action_distribution, value=value, value_hat=value_hat)

        if self.tensorboard_writer is not None:
            # Temporary, for debugging
            with torch.no_grad():
                l1_action, l1_value = F.l1_loss(action_hat, action), F.l1_loss(value_hat, value)
            self.tensorboard_writer.add_scalar('selfmodel/loss_action', loss_action.item(), self.iter)
            self.tensorboard_writer.add_scalar('selfmodel/l1_action', l1_action.item(), self.iter)
            self.tensorboard_writer.add_scalar('selfmodel/loss_value', loss_value.item(), self.iter)
            self.tensorboard_writer.add_scalar('selfmodel/l1_value', l1_value.item(), self.iter)
            log.debug(f'[SelfModel] loss_a: {loss_action:.6f} ({l1_action}), loss_v: {loss_value:.6f} ({l1_value})')

        return loss_action+loss_value

class SelfModelCriterion(nn.Module):
    def __init__(self, dataset, T=2):
        super().__init__()
        # self.crit_action = SoftBCEWithLogitsLoss(T=T)
        self.crit_value = SoftBCEWithLogitsLoss(T=T)
        # TODO: Make lambda coefficient for weights of criterions
        # self.lambda

    def forward(self, action, action_distribution, value, value_hat):
        """
        Computes 2 types of criterion for the World model (observation, reward prediction).

        Parameters
        ----------
        observation_next : tensor of shape (n_samples, dim_observation)
            Ground truth next observation
        observation_next_hat: tensor of shape (n_samples, dim_observation)
            Predicted next observation
        reward_next: tensor of shape (n_samples,)
            Ground truth next reward
        reward_next_hat_prob: tensor of shape (n_samples,)
            Predicted next reward

        Returns
        -------
        loss : tensor of shape (1,)
        """
        # loss_action = self.crit_action(action_hat, action)
        loss_action = -action_distribution.log_prob(action).mean() # Maximize log probability of actions given action distribution
        loss_value = self.crit_value(value_hat, value)
        return loss_action, loss_value

# %%
# Hacking stablebaselines
class DummyLogger(object):
    """Dummy logger for stable baselines"""
    def __init__(self, tensorboard_writer=None):
        self.tensorboard_writer=tensorboard_writer

    def record(self, name, value, *args,**kwargs):
        pass

    def add_scalar(self, name, value, iter=None):
        self.tensorboard_writer.add_scalar(name, value, iter)

class DummyEnv(gym.Env):
    def __init__(self, env, observation_space=None):
        self.action_space = dcopy(env.action_space)
        self.observation_space = dcopy(env.observation_space) if observation_space is None else observation_space

class HackSBModelActorcritic(BaseModel):
    def __init__(self, nn, nn_type, amp=False, timewindow=2):
        super().__init__(nn=nn, amp=amp)
        self.timewindow=timewindow
        self.nn_type=nn_type

    def preprocess(self, observation_history):
        """
        (Explanation here)
        Parameters
        ----------
        observation_history : dict of tensors with shape (Batch, timewindow, dim_observation)

        Returns
        -------
        observation_history_coordinates : tensor with shape (Batch, timewindow*dim_observation*len(dict))
        """
        if self.nn_type=='linear':
            if self.timewindow==1:
                observation_history_concatenated = concat_observation(observation_history) # Dict of shape (Batch, time, 6)
                observation_history_coordinates = observation_history_concatenated.reshape(len(observation_history_concatenated),-1) # shape (Batch, 6*time)

                return observation_history_coordinates
            else:
                observation_history_concatenated = concat_observation(observation_history) # Dict of shape (Batch, time, 6)
                observation_history_coordinates = observation_history_concatenated.reshape(len(observation_history_concatenated),-1) # shape (Batch, 6*time)

                return observation_history_coordinates
        elif self.nn_type=='transformer':
            '''
            observation_history : dict of tensors with shape (Batch, timewindow, dim_observation)

            Returns
            tokens : tensors with shape (Batch, Sequence, 7)

            --------------------------------------------------------------
            Assumes the same number and type of objects in the observation

            Tokens are vectors of dim 7 that corresponds to:
            [x, y, Agent, Goal, Predator, retrieve_action, retrieve_value]

            (Visualization)
            Each batch looks like:

            x, y, A, G, P, action, value
            x  y |1  0  0    0       0   --> for timestep 0
            x  y |0  1  0    0       0
            x  y |0  0  1    0       0
            ----------------------------
            x  y |1  0  0    0       0   --> for timestep -1
            x  y |0  1  0    0       0
            x  y |0  0  1    0       0
            ----------------------------
            0  0  0  0  0    1       0
            0  0  0  0  0    0       1

            '''
            keys = sorted(observation_history.keys()) # Should be [A, G, P] or [A, G1, G2, P1, P2, ...]

            n_batch, n_object_type, n_time, n_object, n_cues = O.__len__(observation_history), 3, observation_history[keys[0]].shape[1], len(keys), 2
            coordinates = np.stack([observation_history[k] for k in keys], axis=1) # shape: (buffer_size, observation_dim)
            coordinates = np.concatenate([observation_history[k] for k in keys], axis=2).reshape(n_batch, -1, 2) # shape: (buffer_size, observation_dim)

            object_indicators = np.zeros((n_batch, n_object, n_object_type+n_cues))
            object_indices = [object_indices_map[k] for k in keys]
            object_indicators[:,range(n_object),object_indices] = 1

            # np.broadcast_to(object_indicators, shape=(n_batch, 3*n_time, 5)) # Why doesn't this work??
            object_indicators = np.tile(object_indicators, reps=(1, n_time, 1))

            features = np.concatenate([coordinates, object_indicators], axis=2)
            retrieve_tokens_ = np.broadcast_to(retrieve_tokens, shape=(n_batch, 2, 7))

            tokens = np.concatenate([features, retrieve_tokens_], axis=1)

            return tokens


class HackSBModelPolicy(HackSBModelActorcritic):
    call_arguments = ['stochastic', 'return_distrib']
    def __init__(self, nn, nn_type, action_space, amp=False, timewindow=2):
        super().__init__(nn=nn, nn_type=nn_type, amp=amp, timewindow=timewindow)
        self.action_space = action_space

    def forward(self, observation_history, stochastic=False, return_distrib=False):
        # Return distribution object
        if return_distrib:
            # Compute additional data (value, log_probs)
            self.nn.eval()
            # observation_history, vectorized_env = self.nn.obs_to_tensor(observation_history)

            with torch.no_grad():
                features = self.nn.extract_features(observation_history)
                latent_pi = self.nn.mlp_extractor.forward_actor(features)
                mean_action = self.nn.action_net(latent_pi)
                action_distribution = get_action_distribution(mean_action, self.nn.log_std.exp().expand_as(mean_action), self.action_space)
            action = action_distribution
        else:
            # action, _states = self.nn.predict(observation_history, deterministic=not stochastic) # Use this for clarity
            action = self.nn._predict(observation_history, deterministic=not stochastic) # Use this for clarity
        return action
    def postprocess(self, action):
        if isinstance(self.action_space, gym.spaces.Box) and isinstance(action, np.ndarray):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

class HackSBModelValue(HackSBModelActorcritic):
    def forward(self, observation_history):
        # No need to call nn.eval() which is implemented within policy.predict()
        self.nn.eval()
        # observation_history, vectorized_env = self.nn.obs_to_tensor(observation_history)
        with torch.no_grad(): value = self.nn.predict_values(obs=observation_history)
        # if not vectorized_env: value = value.squeeze(axis=0)
        return value
    def postprocess(self, value):
        return value


def hijack_memory(memory, model_sb, model):
    """
    Link agent's memory into stablebaselines3 agent.rollout_buffer
    Note that new arrays are replaced into the rollout_buffer.{memory_type} rather than filling within the existing arrays, for speed.

    Parameters
    ----------
    arg : array-like of shape (n_samples,), default=None
        Argument explanation.
        If ``None`` is given, those that appear at least once
        .. versionadded:: 0.18
    """
    # retrieve neat data from memory
    memory = memory['temp']
    observation_history, reward, end, action, value, advantage, return_, log_action_prob = \
    memory['observation_history'], memory['reward'], memory['end'], memory['action'], memory['value'], memory['advantage'], memory['return'], memory['log_action_prob']

    observation_history = model.preprocess(observation_history=observation_history)

    # Synchronize our timepoints to stabebaseline (actorcritic) timepoints
    dict_observation = type(model_sb.observation_space)==gym.spaces.Dict
    observation_history = O.__getitem__(observation_history, slice(None,-1))
    reward, end, action, value, advantage, return_, log_action_prob = reward[1:], end[:-1], action[:-1], value[:-1], advantage[:-1], return_[:-1], log_action_prob[:-1]  # Last value of advantage&return_ is NaN, and will be eliminated,

    # Remove invalid data (between episodes), end is not required actually, since all values (value, advantage, return_) are computed already.
    idx_map = nan_idx_converter(reward)
    observation_history = O.__getitem__(observation_history, idx_map)
    reward, end, action, value, advantage, return_, log_action_prob = reward[idx_map], end[idx_map], action[idx_map], value[idx_map], advantage[idx_map], return_[idx_map], log_action_prob[idx_map]

    # Compute total number valid of samples
    n_samples = len(end) # Number of valid transitions
    model_sb.rollout_buffer.buffer_size=n_samples

    # Connect data
    if dict_observation:
        for k, array in model_sb.rollout_buffer.observations.items():
            model_sb.rollout_buffer.observations[k] = observation_history[k].reshape(n_samples,*array.shape[1:])
    else:
        # model_sb.rollout_buffer.observations = observation_history.reshape(n_samples, *model_sb.rollout_buffer.observations.shape[1:])
        model_sb.rollout_buffer.observations = observation_history.reshape(n_samples, *observation_history.shape[1:]) if model_sb.rollout_buffer.generator_ready else observation_history.reshape(n_samples, 1, *observation_history.shape[1:])

    model_sb.rollout_buffer.actions = action.reshape(n_samples, *model_sb.rollout_buffer.actions.shape[1:])
    model_sb.rollout_buffer.rewards = reward.reshape(n_samples, *model_sb.rollout_buffer.rewards.shape[1:])
    model_sb.rollout_buffer.episode_starts = end.reshape(n_samples, *model_sb.rollout_buffer.episode_starts.shape[1:])
    model_sb.rollout_buffer.values = value.reshape(n_samples, *model_sb.rollout_buffer.values.shape[1:])
    model_sb.rollout_buffer.advantages = advantage.reshape(n_samples, *model_sb.rollout_buffer.advantages.shape[1:])
    model_sb.rollout_buffer.returns = return_.reshape(n_samples, *model_sb.rollout_buffer.returns.shape[1:])
    model_sb.rollout_buffer.log_probs = log_action_prob.reshape(n_samples, *model_sb.rollout_buffer.log_probs.shape[1:])

    # Ready to use
    model_sb.rollout_buffer.full=True

log_actorcritic = logging.getLogger('actorcritic')
def hacked_train(self, n_iter): # Copy-pasted from stablebaselines just to add the log.debug() which prints out the loss value
    """
    Update policy using the currently gathered rollout buffer.
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

    device = U.get_device(self.policy)

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    continue_training = True

    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
        approx_kl_divs = []
        # Do a complete pass on the rollout buffer
        # for rollout_data in self.rollout_buffer.get(self.batch_size):
        for i, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
            # try:
            actions = rollout_data.actions.to(device)
            observations = O.to(rollout_data.observations, device=device)
            advantages = rollout_data.advantages.to(device)
            old_log_prob = rollout_data.old_log_prob.to(device)
            old_values = rollout_data.old_values.to(device)
            returns = rollout_data.returns.to(device)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)
            #
            # if epoch==0 and i==0:
            #     log.info(observations.shape)
            #     import pdb; pdb.set_trace()
            values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
            values = values.flatten()
            # Normalize advantage
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - old_log_prob)
            # if not torch.isfinite(ratio).all(): import pdb; pdb.set_trace()

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            # pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the different between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = old_values + torch.clamp(
                    values - old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Calculate approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html
            with torch.no_grad():
                log_ratio = log_prob - old_log_prob
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                break

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Check nan in grad. (Comes from ratio being inf)
            if not all([torch.isfinite(p.grad).all() for p in self.policy.parameters()]): # any inf or nan
                # We could try eliminating the data samples that caused inf/nan, but once they're inside the graph pytorch computes every gradient which causes parameter.grad==nan anyways.
                continue
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Logging
            self.logger.add_scalar('actorcritic/entropy_loss', entropy_loss.item(), n_iter)
            self.logger.add_scalar('actorcritic/policy_gradient_loss', policy_loss.item(), n_iter)
            self.logger.add_scalar('actorcritic/value_loss', value_loss.item(), n_iter)
            self.logger.add_scalar('actorcritic/approx_kl', approx_kl_div, n_iter)
            self.logger.add_scalar('actorcritic/clip_fraction', clip_fraction, n_iter)
            self.logger.add_scalar('actorcritic/loss', loss.item(), n_iter)
            if hasattr(self.policy, "log_std"): self.logger.add_scalar("actorcritic/std", torch.exp(self.policy.log_std).mean().item(), n_iter)
            n_iter+=1

            log_actorcritic.debug(f'[Node: actorcritic][Epoch: {epoch}/{self.n_epochs}][Loss: {loss.item():.7f}]')
            # except Exception as e:
            #     log_actorcritic.info(e)
            #     import pdb; pdb.set_trace()
        if not continue_training:
            break

    self._n_updates += self.n_epochs

class HackSBTrainer(object):
    def __repr__(self):
        return f'HackSBTrainer with {self.model_sb}'
    def __init__(self, model, model_sb, nn_sb):
        self.model=model # BaseModel class
        self.model_sb=model_sb # Stablebaselines model class
        self.nn=nn_sb
        self.memory=None
        self.iter=np.array([0]) # Use numpy array for the self.model_sb.train() to modify self.iter
        self.n_train=0
    def train(self):
        hijack_memory(self.memory, self.model_sb, self.model)
        self.n_train+=1
        self.model_sb.train(self.iter)
        self.model_sb.rollout_buffer.full=False # Don't train twice. Just increase the epoch.
        print(self.iter)
        # self.iter+=len(self.model_sb.rollout_buffer.returns)//self.model_sb.batch_size*self.model_sb.n_epochs

def hack_stablebaselines(TrainerType, env, cfg, tensorboard_writer):
    # Don't use observation_preprocessor when hacking stablebaselines. It gets tricky to handle observation_space
    # if cfg.agent.actorcritic.timewindow==1:
    #     env_dummy = DummyEnv(env)
    #     if type(env_dummy.observation_space)==gym.spaces.Dict:
    #         policy_type = 'MultiInputPolicy'
    #     else:
    #         policy_type = 'MlpPolicy'
    # else:

    observation_space = gym.spaces.utils.flatten_space(gym.spaces.Tuple([env.observation_space]*cfg.agent.actorcritic.timewindow)) # Concatenating multistep observation
    env_dummy = DummyEnv(env, observation_space=observation_space)
    if cfg.agent.actorcritic.type == 'linear':
        policy_type = 'MlpPolicy'
        policy_kwargs={}
    elif cfg.agent.actorcritic.type == 'transformer':
        policy_type = TransformerPolicy
        policy_kwargs = dict(
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs=dict(features_dim=cfg.agent.actorcritic.features_dim, nhead=cfg.agent.actorcritic.nhead, dim_feedforward=cfg.agent.actorcritic.dim_feedforward, dropout=cfg.agent.actorcritic.dropout),
            separate_input=cfg.agent.actorcritic.separate_input
        )
    policy_kwargs['net_arch'] = [{'pi':OmegaConf.to_container(cfg.agent.actorcritic.layers), 'vf':OmegaConf.to_container(cfg.agent.actorcritic.layers)}]

    actorcritic = TrainerType(policy_type, env_dummy,
            learning_rate=cfg.agent.actorcritic.cfg_train.lr, batch_size=cfg.agent.actorcritic.cfg_train.batch_size, n_epochs=cfg.agent.actorcritic.cfg_train.epoch,
            policy_kwargs=policy_kwargs,
            verbose=1) # stable baseline3 agent
    actorcritic._logger = DummyLogger(tensorboard_writer=tensorboard_writer)
    # actorcritic._logger = tensorboard_writer

    # Define nn
    nn_actorcritic = actorcritic.policy

    # Define model
    model_actorcritic = {}
    model_actorcritic['policy'] = HackSBModelPolicy(nn=nn_actorcritic, nn_type=cfg.agent.actorcritic.type, timewindow=cfg.agent.actorcritic.timewindow, action_space=env_dummy.action_space)
    model_actorcritic['value'] = HackSBModelValue(nn=nn_actorcritic, nn_type=cfg.agent.actorcritic.type, timewindow=cfg.agent.actorcritic.timewindow)

    # Define trainer
    trainer_actorcritic = HackSBTrainer(model=model_actorcritic['policy'], model_sb=actorcritic, nn_sb=nn_actorcritic)

    # Additional hacks for stable computation
    actorcritic.train = hacked_train.__get__(actorcritic, type(actorcritic))

    return nn_actorcritic, model_actorcritic, trainer_actorcritic

# %%
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Tokens are vectors of dim 7 that corresponds to:
# [x, y, Agent, Goal, Predator, retrieve_action, retrieve_value]
retrieve_action = np.array([[0,0,0,0,0,1,0]])
retrieve_value = np.array([[0,0,0,0,0,0,1]])
retrieve_tokens = np.concatenate([retrieve_action,retrieve_value], axis=0)
object_types = sorted(['A','G','P'])
object_indices_map = {object_type:i for i, object_type in enumerate(object_types)}

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim: int = 1024,
        nhead: int=1,
        dim_feedforward: int=2048,
        dropout: float=0.1
    ):
        super().__init__(observation_space, features_dim)

        self.embedder = nn.Linear(7, features_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

    def forward(self, tokens):
        '''
        features: tensor of shape (Batch, sequence, dimension)
        and the last 2 sequences correspond to action, value

        '''
        # Embedding
        n_batch, n_sequence, n_dim = tokens.shape
        embedding = self.embedder(tokens.reshape(-1, n_dim)).reshape(n_batch, n_sequence, -1)

        # Positional encoding?

        features_transformed = self.transformer(embedding) # (Bathc, sequence, dimension)
        features_actionvalue = features_transformed[:,-2:,:].reshape(n_batch, 2*self.features_dim) # (Batch, 2*features_dim)
        return features_actionvalue

    def encode_position(self, tokens):
        pass

class TransformerMLPExtractor(nn.Module):
    '''
    MLP Extractor identical to stable_baselines3 default,
    with an additional option to use parts/all of the features for pi/vf
    '''
    def __init__(self, features_dim, net_arch, separate_input):
        super().__init__()

        self.features_dim=features_dim
        self.separate_input=separate_input

        # Policy net
        def make_layers(n_hidden_list):
            layers = []
            for i, n_neurons in enumerate(n_hidden_list):
                if i==0:
                    if self.separate_input:
                        layers.append(nn.Linear(self.features_dim, n_neurons))
                    else:
                        layers.append(nn.Linear(2*self.features_dim, n_neurons))
                else:
                    layers.append(nn.Linear(last_n_neurons, n_neurons))
                layers.append(nn.Tanh())
                last_n_neurons = n_neurons
            return layers

        policy_layers = make_layers(net_arch[0]['pi'])
        value_layers = make_layers(net_arch[0]['vf'])

        self.policy_net = nn.Sequential(*policy_layers)
        self.value_net = nn.Sequential(*value_layers)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        def compute_latent_dim(layers):
            if len(layers)!=0:
                return layers[-2].out_features
            elif self.separate_input:
                return self.features_dim
            else:
                return 2*self.features_dim

        # self.latent_dim_pi = net_arch[0]['pi'][-1] if len(policy_layers)!=0 else self.features_dim
        # self.latent_dim_vf = net_arch[0]['vf'][-1] if len(value_layers)!=0 else self.features_dim
        self.latent_dim_pi = compute_latent_dim(policy_layers)
        self.latent_dim_vf = compute_latent_dim(value_layers)

    def forward(self, features):
        '''
        features: shape of (Batch, 2*features_dim)
        '''
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        action = self.policy_net(features[:, :self.features_dim]) if self.separate_input else self.policy_net(features)
        return action

    def forward_critic(self, features):
        value = self.value_net(features[:, self.features_dim:]) if self.separate_input else self.value_net(features)
        return value

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule ,
        features_dim = 1024,
        separate_input = True,
        *args,
        **kwargs,
        ):

        self.separate_input=separate_input

        super().__init__(observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,)

    def _build_mlp_extractor(self):
        self.mlp_extractor = TransformerMLPExtractor(features_dim=self.features_dim, net_arch=self.net_arch, separate_input=self.separate_input)
