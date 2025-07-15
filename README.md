# Dualpolicy_selfmodel
Code for "Dual policy as self-model for planning, Journal of Korean Institute of Intelligent Systems (2024)"([Arxiv](https://arxiv.org/abs/2306.04440), [Journal](https://dx.doi.org/10.5391/JKIIS.2024.34.1.15))

This code is licensed under the terms of the CC BY-NC-ND 4.0 license.
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Please cite:

    @article{yoo2024dual,
    title={Dual Policy as Self-Model for Planning},
    author={Yoo, Jaesung and de la Torre, Fernanda and Yang, Guangyu Robert},
    journal={Journal of Korean Institute of Intelligent Systems},
    volume={34},
    number={1},
    pages={15--24},
    year={2024},
    doi = {https://doi.org/10.5391/JKIIS.2024.34.1.15},
    }

And star(★) the repository! Thank you!

# Code philosophy
Powered by [hydra](https://hydra.cc/docs/intro/) and prototype version of [hideandseek](https://pypi.org/project/hideandseek/).

## If you're new to hydra & omegaconf
You can read [hydra](https://hydra.cc/docs/intro/) documentation, but it might be overwhelming at first. I recommend trying out `scripts/tutorial/hydra_jupyter_sample.ipynb` to understand how configuration management works first, and then try running `scripts/tutorial/hydra_sample.py` in the shell to see the [job management feature](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/), [multirun hyperparameter sweep](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) functions of hydra.

# Results
## 1. Dual policy agent achieves similar performance to shared policy agent with better training stability

```python
nohup python run.py +exp=assumption6 & # Use your own hydra multirun launcher for parallelism
```

```python
python scripts/figures/reward_plot.py exp_name=assumption6
```
## 2. Dual policy agent have faster planning inference with smaller distilled networks

```python
nohup python run.py +exp=cost4 & # Use your own hydra multirun launcher for parallelism
```

```python
python scripts/figures/cost.py exp_name=cost4
```

## 3. Dual policy enhances exploration

```python
nohup python run.py +exp=mapsize3 & # Use your own hydra multirun launcher for parallelism
```

```python
python scripts/figures/reward_plot.py exp_name=mapsize3
```

## 4. Dual policy learns comprehensive understanding of the agent’s behaviors

```python
nohup python run.py +exp=assumption_innatetrigger4 & 
```

```python
python scripts/figures/reward_plot.py exp_name=assumption_innatetrigger4
```



