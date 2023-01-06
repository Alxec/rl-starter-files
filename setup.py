"""Sets up the rl-starter-files module."""

from setuptools import setup

setup(
    name='rl-starter-files',
    version='1.0.0',
    keywords='memory, environment, agent, rl, gymnasium',
    url='https://github.com/Alxec/rl-starter-files',
    description='RL starter files in order to immediatly train, visualize and evaluate an agent without writing any line of code',
    packages=['rl-starter-files'],
    install_requires=[
        'torch-ac>=1.4.0',
        'minigrid',
        'tensorboardX>=1.6',
        'numpy>=1.3',
        'gymnasium>=0.26'
    ]
)
