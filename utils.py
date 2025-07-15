import os
from copy import deepcopy
import itertools as it
import pickle

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

# %%
def get_device(model):
    return next(model.parameters()).device

def reproducible_worker_dict():
    '''Generate separate random number generators for workers,
    so that the global random state is not consumed,
    thereby ensuring reproducibility'''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {'worker_init_fn': seed_worker, 'generator': g}

def save_pickle(obj: str, path: str = None, protocol: int = None):
    '''Save object as Pickle file to designated path.
    If path is not given, default to "YearMonthDay_HourMinuteSecond.p" '''
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
        warnings.warn(f'Be sure to specify specify argument "path"!, saving as {path}...')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle(path: str):
    '''Load Pickle file from designated path'''
    with open(path, 'rb') as f:
        return pickle.load(f)

def merge_dict(ld):
    '''merge list of dicts
    into dict of lists
    ld: list of dicts'''

    # keys = sorted(list(set([d.keys() for d in ld])))
    keys = sorted(list(set(it.chain(*[d.keys() for d in ld]))))
    d_merged = {key:[] for key in keys}
    for d in ld:
        for k, v in d.items():
            d_merged[k].append(v)
    return d_merged

def reverse_dict(d):
    '''
    Reverse key:value pair of dict into value:key.
    Thus, values in dict must be hashable
    '''
    d_ = {}
    for k, v in d.items():
        d_[v]=k
    return d_

def ld_to_array(ld):
    '''
    converts list of dicts into dict of np.ndarrays
    '''
    if isinstance(ld[0], dict):
        return {k:np.array(v) for k, v in merge_dict(ld).items()}
    else:
        return np.array(ld)

def equal(lst):
    '''return True if all elements in iterable is equal'''
    lst_inst = iter(lst)
    try:
        val = next(lst_inst)
    except StopIteration:
        return True

    for v in lst_inst:
        if v!=val:
            return False
    return True

# %%
def seed(random_seed, strict=False):
    '''

    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if strict:
        # Following is verbose, but just in case.
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # deterministic cnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %%
class Path(str):
    '''
    Joins paths by . syntax
    (Want to use pathlib.Path internally, but currently inherit from str)

    Parameters
    ----------
    path: str (default: '.')
        Notes the default path. Leave for default blank value which means the current working directory.
        So YOU MUST NOT USE "path" AS ATTRIBUTE NAME, WHICH WILL MESS UP EVERYTHING

    Example
    -------
    >>> path = Path('C:/exp')
    >>> path
    path: C:/exp

    >>> path.DATA = 'CelebA'
    >>> path
    path: C:/exp
    DATA: C:/exp/CelebA

    >>> path.PROCESSED = 'processed'
    >>> path.PROCESSED.M1 = 'method1'
    >>> path.PROCESSED.M2 = 'method2'
    >>> path
    path: C:/exp
    DATA: C:/exp/CelebA
    PROCESSED: C:/exp/processed

    >>> path.PROCESSED
    M1: C:/exp/processed/method1
    M2: C:/exp/processed/method2
    -------

    '''
    def __init__(self, path='.'):
        self.path=path

    def __repr__(self):
        return f'Path({self.path})'

    def __call__(self, indent=0):
        '''Print out current path, and children'''
        for name, directory in self.__dict__.items():
            if name != 'path':
                print(' '*indent+name+': '+str(directory))
                if type(directory) == Path:
                    directory(indent+2)
        # print('\n'.join([key+': '+str(value) for key, value in self.__dict__.items()]))

    def __str__(self):
        return self.path

    def __setattr__(self, key, value):
        # super(Path, self).__setattr__(key, self / value) # self.joinpath(value)
        if hasattr(self, 'path'):
            assert key != 'path', '"path" is a predefined attribute and must not be used. Use some other attribute name'
            super(Path, self).__setattr__(key, Path(os.path.join(self.path, value)))
        else:
            super(Path, self).__setattr__(key, value)

    def join(self, *args):
        return Path(os.path.join(self.path, *args))

    def makedirs(self, exist_ok=True):
        '''Make directories of all children paths
        Be sure to define all folders first, makedirs(), and then define files in Path(),
        since defining files before makedirs() will lead to creating directories with names of files.
        It is possible to ignore paths with "." as all files do, but there are hidden directories that
        start with "." which makes things complicated. Thus, defining folders -> makedirs() -> define files
        is recommended.'''
        for directory in self.__dict__.values():
            if directory != '':
                os.makedirs(str(directory), exist_ok=exist_ok)
                if type(directory) == Path:
                    directory.makedirs(exist_ok=exist_ok)

    def clear(self, ignore_errors=True):
        '''Delete all files and directories in current directory'''
        for directory in self.__dict__.values():
            shutil.rmtree(directory, ignore_errors=ignore_errors)

    def listdir(self, join=False, isdir=False, isfile=False):
        if isdir or isfile:
            return _os.listdir(self.path, join=join, isdir=isdir, isfile=isfile)
        else:
            if join:
                return [os.path.join(self.path, p) for p in os.listdir(self.path)]
            else:
                return os.listdir(self.path)

class Counter(object):
    def __init__(self, keys):
        self.keys = keys
        self.map = {key:i for i, key in enumerate(keys)}
        self.sequence = []

    def count(self, key):
        self.sequence.append(self.map[key])

    def reset(self):
        self.sequence = []

class ValueTracker(object):
    """ ValueTracker."""

    def __repr__(self):
        return f'<ValueTracker>\nx: {self.x}\ny: {self.y}'

    def __init__(self):
        self.reset()

    def __len__(self):
        return len(self.y)

    def __iadd__(self, other):
        self.x.extend(other.x)
        self.y.extend(other.y)
        self.label.extend(other.label)
        self.n_step += len(other.x)
        return self

    def __add__(self, other):
        self = deepcopy(self)
        self.x.extend(other.x)
        self.y.extend(other.y)
        self.label.extend(other.label)
        self.n_step += len(other.x)
        return self

    def reset(self):
        self.x = []
        self.y = []
        self.label = []
        self.n_step = 0

    def numpy(self):
        return np.array(self.x), np.array(self.y), np.array(self.label)

    def step(self, x, y, label=None):
        if hasattr(x, '__len__'):
            assert hasattr(y, '__len__')
            assert len(x)==len(y)
            self.x.extend(x)
            self.y.extend(y)
            if label != None:
                assert len(y)==len(label)
                self.label.extend(label)
            self.n_step += len(x)

        else:
            self.x.append(x)
            self.y.append(y)
            if label != None:
                self.label.append(label)
            self.n_step += 1

    def plot(self, w=9, color='tab:blue', ax=None):
        x = np.array(self.x)
        y = np.array(self.y)
        y_smooth = moving_mean(y, w)
        if ax==None:
            ax = plt.gca()
        ax.plot(x, y, color=color, alpha=0.4)
        ax.plot(x, y_smooth, color=color)
        return ax

    def mean(self):
        return np.mean(self.y)
    def min(self):
        return np.min(self.y)
    def max(self):
        return np.max(self.y)

class AverageMeter(object):
    """Computes and stores the average and current value
    Variables
    ---------
    self.val
    self.avg
    self.sum
    self.count
    """
    # TODO: maybe keep track of each values? or just merge with valuetracker?

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def step(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# %%
def moving_mean(x, w):
    odd = bool(w%2)
    edge = w//2
    if odd:
        x = np.pad(x, w//2+1, mode='edge')
        x = np.cumsum(x).astype(np.float64)
        x = (x[w:] - x[:-w])/w
        return x[:-1]
    else:
        x = np.pad(x, w//2, mode='edge')
        x = np.cumsum(x).astype(np.float64)
        x = (x[w:] - x[:-w])/w
        return x

def nanstd(input):
    '''
    Disregard NaNs to compute std.

    Note that torch.std() is an overloaded function, and it's hard to customize.
    Since we just need a simple float as a result, not an array,
    Just kept it simple.
    '''
    return torch.std(input[~input.isnan()])

# %%
# Plot utility
def compute_margin(ymin, ymax, margin_ratio):
    margin = margin_ratio / (1-margin_ratio*2) * (ymax - ymin)
    return (ymin-margin, ymax+margin)
