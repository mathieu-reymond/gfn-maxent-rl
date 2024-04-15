from numpy.random import default_rng

from gfn_maxent_rl.envs.seh.env import SEHEnvironment
from gfn_maxent_rl.envs.seh.rewards import RewardProxy


def get_seh_env(
    name,
    num_envs=1,
    rng=default_rng(),
    seed=0,
    **kwargs
):

    env = SEHEnvironment(num_envs, seed=seed)

    infos = {'metadata': kwargs}

    return (env, infos)
