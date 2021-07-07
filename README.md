# Context-Adaptive Reinforcement Learning using Unsupervised Learning of Context Variables

## Installation

Clone Repository and change directory:

```bash
git clone git@github.com:eghbalz/CarlaRL.git
cd carla
```

Create conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate carla
```

Install project in the activated environment

```bash
python setup.py develop
```

## Manual Control
To manually play the environment run the following command
```
python manual_control.py
```

## Create datasets for VAE and Context Classification
create db.path in `datasets` folder and put the path to each dataset. for example:
```
carla45fully48px:/path/to/data/carla45fully48pxdb
carla45fully48px_val:/path/to/data/carla45fully48px_valdb
carla45fully48px_tst:/path/to/data/carla45fully48px_tstdb
```
