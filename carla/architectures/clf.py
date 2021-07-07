"""
Created by Hamid Eghbal-zadeh at 05.02.21
Johannes Kepler University of Linz
"""


from torch import nn
from typing import TypeVar

Tensor = TypeVar('torch.tensor')

'''
context classifier
'''


class LinearContextClf(nn.Module):
    def __init__(self, n_latent, n_context):
        super(LinearContextClf, self).__init__()
        self.clf = nn.Linear(n_latent, n_context)

    def forward(self, x):
        out = self.clf(x)
        return out

    def init_weights(self, init_fn):
        self.clf.apply(init_fn)


class MLPContextClf(nn.Module):
    def __init__(self, n_latent, n_context):
        super(MLPContextClf, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(n_latent, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), )
        self.clf = nn.Linear(128, n_context)

    def forward(self, x):
        out = self.features(x)
        out = self.clf(out)
        return out

    def init_weights(self, init_fn):
        self.clf.apply(init_fn)
        self.features.apply(init_fn)


