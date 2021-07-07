"""
Created by Hamid Eghbal-zadeh at 05.02.21
Johannes Kepler University of Linz
"""

import os

def check_dir(directory):
    if not os.path.exists(directory):
        print('{} not exist. calling mkdir!'.format(directory))
        os.makedirs(directory)
