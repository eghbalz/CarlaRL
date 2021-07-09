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

## Citation

```
@InProceedings{pmlr-v148-eghbal-zadeh21a,
  title     =  {Context-Adaptive Reinforcement Learning using Unsupervised Learning of Context Variables},
  author    =  {Eghbal-zadeh, Hamid and Henkel, Florian and Widmer, Gerhard},
  booktitle =  {NeurIPS 2020 Workshop on Pre-registration in Machine Learning},
  pages     =  {236--254},
  year      =  {2021},
  editor    =  {Bertinetto, Luca and Henriques, João F. and Albanie, Samuel and Paganini, Michela and Varol, Gül},
  volume    =  {148},
  series    =  {Proceedings of Machine Learning Research},
  month     =  {11 Dec},
  publisher =  {PMLR},
  pdf       =  {http://proceedings.mlr.press/v148/eghbal-zadeh21a/eghbal-zadeh21a.pdf},
  url       =  {http://proceedings.mlr.press/v148/eghbal-zadeh21a.html}
}
```
