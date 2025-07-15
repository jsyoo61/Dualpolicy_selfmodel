'''
Recommend running the following on the shell script as a tutorial.
`python train.py` # basic functionality
`python train.py -m run.random_seed=0,1,2,3,4` # multirun sweep
`python train.py -m "run.random_seed=range(0,5)` # multirun sweep with predefined hydra functions
`python train.py +new_variable=0` # Inserting new configs/config_groups
`python train.py -m +new_variable=0,1,2,3,4` # Insert new config & sweep
`python train.py hydra.run.dir=outputs/custom_directory` # Specifying custom directory by overriding (hidden) hydra configuration
`python train.py -m hydra.sweep.dir=multirun/custom_directory run.random_seed=0,1,2,3,4` # Specifying custom multirun directory by overriding (hidden) hydra configuration
`python train.py -m hydra.sweep.dir=multirun/custom_directory hydra.sweep.subdir=run_${run.random_seed}` # Specifying custom multirun directory & subdir by overriding (hidden) hydra configuration
`python train.py -m hydra/launcher=joblib hydra.launcher.n_jobs=2 "run.random_seed=range(0,10)"` # Need to run `pip install hydra-joblib-launcher` first! Multirun sweep with parallel processes & n_job limit
`python train.py hydra.debug=True"` # Run using different logging (print) levels
'''

import hydra
from omegaconf import OmegaConf, DictConfig
import logging

import os

log = logging.getLogger(__name__)

# %%
@hydra.main(config_path='config', config_name='train_cfg')
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.debug(os.getcwd())

if __name__=='__main__':
    main()
