"""
Created by Hamid Eghbal-zadeh at 05.02.21
Johannes Kepler University of Linz
"""
"""
Created by Hamid Eghbal-zadeh at 10.09.20
Johannes Kepler University of Linz
"""

from torch.utils.data import Sampler

class SeqSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)