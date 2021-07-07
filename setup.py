
import setuptools

from setuptools import setup


setup(
    name='carla',
    version='0.1dev',
    description='Context-Adaptive Reinforcement Learning using Unsupervised Learning of Context Variables',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: ReinforcementLearning",
    ],
    author='Florian Henkel, Hamid Eghbal-zadeh',
)
