

import hydra
from omegaconf import OmegaConf, DictConfig

# %%
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir='C:\\Users\\jsyoo\\Mirror\\Metaconscious Lab\\Research\\SelfModel\\code_SelfModel\\config', job_name='debug')

# %%
overrides = []
cfg = hydra.compose(config_name='train_cfg', overrides=overrides)

# %%
print(OmegaConf.to_yaml(cfg))

# %%
import logging
import sys
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))

# %%
log.setLevel(logging.DEBUG)
log.info('info level message')
log.debug('debug level message')

log.setLevel(logging.INFO) # May set to logging.DEBUG
log.info('info level message')
log.debug('debug level message')
