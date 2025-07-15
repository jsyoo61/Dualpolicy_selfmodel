'''


'''
import logging
import os
import shutil
from copy import deepcopy as dcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D

import utils as U

__all__ = [
'Node'
]

# %%
log = logging.getLogger(__name__)

# %%
class Trainer:
    '''Local Node for training'''
    def __init__(self, nn: nn.Module=None, memory=None, model=None, DatasetClass=None, OptimizerClass = None, cfg_train={}, CriterionClass: nn.Module=None, model_dir=None, node_dir=None, validation=None, earlystopper=None, name: str='default', verbose=True, amp=False, reproduce=True, tensorboard_writer=None):
        """
        (Explanation here)
        Parameters
        ----------
        nn : torch.nn.Module object

        memory: dict of deque objects

        DatasetClass: torch.utils.data.Dataset class
            This dataset is used for training.
            Recommended return format from dataset is: {'x': x, 'y': y}
            where 'x' is the input, and 'y' is the target values.

        OptimizerClass: torch.optim.Optimizer class

        cfg_train: dict-like object which contains
            lr
            batch_size
            weight_decay (optional)
            patience (optional)

        criterion: torch.nn.Module
            used to compute loss function

        Examples
        --------
        >>> kwargs = {'nn': nn, 'dataset': train_dataset, 'validation': None, 'cfg_train': cfg.train,
            'criterion': criterion, 'model_dir': path['nn'], 'node_dir': path['node'], 'verbose': True, 'amp': True}
        """
        '''
        :param validation: dict of Validation objects


        Recommended way of making hs.node.Node object is like the following:

            node = hs.node.Node(**kwargs)
        '''
        # Store configurations
        self.nn = nn
        self.memory = memory
        self.model = model
        self.DatasetClass = DatasetClass
        self.OptimizerClass = OptimizerClass if OptimizerClass is not None else optim.Adam
        self.cfg_train = dcopy(dict(cfg_train))
        self.CriterionClass = CriterionClass
        self.model_dir = model_dir
        self.node_dir = node_dir
        # self.validation = validation if issubclass(type(validation), dict) or validation is None else V.VDict({'default': validation}) # wrap with VDict if single validation object is given.
        self.validation = validation
        self.earlystopper = earlystopper
        self.name = name
        self.verbose = verbose
        self.amp = amp
        self.reproduce = reproduce
        self.tensorboard_writer = tensorboard_writer

        # Initializations
        self.log = logging.getLogger(f'{__name__}.{self.name}')
        # self.logger = None if verbose else
        self._tensorboard = tensorboard_writer is not None

        if self.model_dir is not None:
            if os.path.exists(self.model_dir): log.warning(f'path: {self.model_dir} exists. Be careful')
            os.makedirs(self.model_dir, exist_ok=True)

        if self.node_dir is not None:
            if os.path.exists(self.node_dir): log.warning(f'path: {self.node_dir} exists. Be careful')
            os.makedirs(self.node_dir, exist_ok=True)

        if self.validation is None and self.earlystopper is not None: log.warning('validation is None but earlystopper is given. earlystopper will be ignored')
        if self.earlystopper is not None:
            assert earlystopper.target_validation in self.validation.keys(), 'earlystopper.target_valiation not provided in validation'

        self.train_meter = U.AverageMeter() # Tracks loss per epoch
        self.loss_tracker = U.ValueTracker() # Tracks loos over all training
        self.reset()

    def __repr__(self):
        return f'Trainer object <name: {self.name}>'

    def reset(self):
        self.iter = 0
        self.n_batch=0

    def print(self, content):
        if self.verbose:
            print(content)
        else:
            self.log.info(content)

    def generate_dataloader(self):
        # Generate dataset
        memory = self.memory['temp']
        self.dataset = self.DatasetClass(memory=memory, model=self.model, convert_tensor=True)

        # Generate dataloader
        reproduce_kwargs = U.reproducible_worker_dict() if self.reproduce else {}
        drop_last = len(self.dataset) % self.cfg_train['batch_size'] == 1 # To avoid batch normalization layers from raising exceptions

        self.loader = D.DataLoader(self.dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=drop_last, **reproduce_kwargs)

    def train(self, epoch=None, new_op=False, no_val=True, step=None, reset_loss_tracker=False):
        '''
        Trains the model with the specified duration.
        '''
        # Training duration check
        assert epoch is None or step is None, f'only one of epoch or step can be specified. received epoch: {epoch}, step: {step}'
        if step is None:
            horizon = 'epoch'
            if epoch is None: # When neither epoch or step are specified
                assert 'epoch' in self.cfg_train, 'key "epoch" must be provided in cfg_train, or the argument "epoch" must be provided \
                                                when argument "step" is not specified.'
                epoch = self.cfg_train['epoch']
            log.debug(f'[Node: {self.name}] train for {epoch} epochs')
        else:
            assert epoch is None, f'Either epoch or step must be specified. Received epoch: {epoch}, step: {step}'
            horizon = 'step'
            step = step
            log.debug(f'[Node: {self.name}] train for {step} steps')

        # Prepare training
        self.nn.train()
        self._device = U.get_device(self.nn)
        if reset_loss_tracker: self.loss_tracker.reset()
        self.generate_dataloader()
        self.criterion = self.CriterionClass(dataset=self.dataset).to(self._device)

        '''
        There may be one or more loaders, but self.loader is the standard of synchronization
        Either return multiple values from dataset, or modify self.forward to use other loaders
        '''

        if self.validation is not None:
            if 'cv_step' not in self.cfg_train:
                self.print('Node.validation given, but "cv_step" not specified in cfg_train. Defaults to 1 epoch')
                self.cfg_train['cv_step'] = len(self.loader)
            self.validation.reset()
            self.validate() # initial testing

        # Make new optimizer
        if new_op or not hasattr(self, 'op'):
            if not hasattr(self, 'op') and not new_op:
                self.log.warning("new_op=False when there's no pre-existing optimizer. Ignoring new_op...")
            # Weight decay optional
            self.op = self.OptimizerClass(self.nn.parameters(), lr=self.cfg_train['lr'], weight_decay=self.cfg_train['weight_decay']) if 'weight_decay' in self.cfg_train \
                        else self.OptimizerClass(self.nn.parameters(), lr=self.cfg_train['lr'])

        # Train loop
        if horizon == 'epoch':
            self._update_epoch(T=epoch, no_val=no_val)
        elif horizon=='step':
            self._update_step(T=step, no_val=no_val)

        # TODO: Return criterion back to its original device, meaning we have to store its previous device info
        self.criterion = self.criterion.cpu()
        return self.loss_tracker

    def _update_epoch(self, T, no_val):

        _iter = 0
        for epoch in range(1, T+1):
            self.train_meter.reset()
            for batch_i, data in enumerate(self.loader, 1):
                self.iter += 1
                _iter += 1
                loss = self._update(data)
                self.log.debug(f'[Node: {self.name}][iter_sum: {self.iter}][Epoch: {epoch}/{T}][Batch: {batch_i}/{len(self.loader)}][Loss: {loss:.7f} (Avg: {self.train_meter.avg:.7f})]')
                self.loss_tracker.step(self.iter, loss)
                if self._tensorboard: self.tensorboard_writer.add_scalar(f'{self.name}/Loss', loss, self.iter)

                # Validation
                if (self.validation is not None) and (not no_val) and (_iter % self.cfg_train['cv_step']==0):
                    patience_end = self.validate()
                    if patience_end: # If patience has reached, stop training
                        self.log.debug('Patience met, stopping training')
                        return None # return None since double break is impossible in python

    def _update_step(self, T, no_val):
        if hasattr(self, '_loader_inst'): del self._loader_inst

        for _iter in range(1, T+1):
            # Get Data
            try:
                if hasattr(self, '_loader_inst'):
                    data = next(self._loader_inst)
                    self.n_batch += 1
                else:
                    raise StopIteration
            except StopIteration as e:
                self.train_meter.reset()
                self._loader_inst = iter(self.loader)
                data = next(self._loader_inst)
                self.n_batch = 1

            self.iter += 1
            loss = self._update(data)
            self.log.debug(f'[Node: {self.name}][iter_sum: {self.iter}][Iter: {_iter}/{T}][Batch: {self.n_batch}/{len(self.loader)}][Loss: {loss:.7f} (Avg: {self.train_meter.avg:.7f})]')

            self.loss_tracker.step(self.iter, loss)
            if self._tensorboard: self.tensorboard_writer.add_scalar(f'{self.name}/Loss', loss, self.iter)

            # Validation
            if (self.validation is not None) and (not no_val) and (_iter % self.cfg_train['cv_step']==0):
                patience_end = self.validate()
                if patience_end: # If patience has reached, stop training
                    self.log.debug('Patience met, stopping training')
                    return None # return None since double break is impossible in python

    def _update(self, data):
        '''
        Pseudo function to support amp (automatic mixed precision)
        '''
        if self.amp:
            # Mixed precision for acceleration
            with torch.autocast(device_type=self._device.type):
                return self.update(data)
        else:
            return self.update(data)

    def update(self, data):
        """
        - Perform single update (forward/backward pass + gradient descent step) with the given data.
        - Store loss in self.train_meter
        - This is where a lot of errors happen, so there's a pdb to save time.
          When there's error, use pdb to figure out the shape, device, tensor dtype, and more.

        Parameters
        ----------
        data : tuple, list, or dict of tensors (Batch of data)
            This is received from a torch.utils.Data.DataLoader
            Depending on the given format, the data is fed to the forwad pass

        Returns
        -------
        loss : float
        """
        try:
            loss, N = self._forward(data)

            self.op.zero_grad()
            loss.backward()
            self.op.step()

            loss = loss.item()
            self.train_meter.step(loss, N)
            return loss

        except Exception as e:
            log.warning(e)
            import pdb; pdb.set_trace()

    def _forward(self, data):
        '''
        Pseudo function to support different type of "data" variable
        '''
        # When data is given as a tuple/list
        if type(data) is tuple or type(data) is list:
            data = [x.to(self._device) for x in data]
            N = len(data[0]) # number of data in batch
            loss = self.forward(*data)

        # When data is given as a dict
        elif type(data) is dict:
            data = {key: value.to(self._device) for key, value in data.items()}
            N = len(next(iter(data))) # number of data in batch
            loss = self.forward(**data)

        else:
            raise Exception(f'return type from dataset must be one of [tuple, list, dict], received: {type(data)}')
        return loss, N

    def forward(self, x, y):
        """
        Forward pass. Receive data and return the loss function.
        Inherit Node and define new forward() function to build custom forward pass.

        May return tuple or dictionary, whichever will be feeded to criterion.
        y_hat = self.nn(x)
        loss = self.criterion(y,y_hat)
        return loss
        """
        pass
