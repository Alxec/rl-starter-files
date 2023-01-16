import gymnasium as gym
from minigrid.wrappers import *

wrappers = {
    "ReseedWrapper": ReseedWrapper,
    "ActionBonus": ActionBonus,
    "StateBonus": StateBonus,
    "ImgObsWrapper": ImgObsWrapper,
    "OneHotPartialObsWrapper": OneHotPartialObsWrapper,
    "RGBImgObsWrapper": RGBImgObsWrapper,
    "RGBImgPartialObsWrapper": RGBImgPartialObsWrapper,
    "RGBImgPartialObsWrapper_HD": RGBImgPartialObsWrapper_HD,
    "FullyObsWrapper": FullyObsWrapper,
    "DictObservationSpaceWrapper": DictObservationSpaceWrapper,
    "FlatObsWrapper": FlatObsWrapper,
    "ViewSizeWrapper": ViewSizeWrapper,
    "DirectionObsWrapper": DirectionObsWrapper,
    "SymbolicObsWrapper": SymbolicObsWrapper         
}


def make_env(env_key, seed=None, wrapper=None, render_mode=None, **kwargs):
    if wrapper:
        env = wrappers[wrapper](gym.make(env_key, render_mode=render_mode), **kwargs)
    else:
        env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
