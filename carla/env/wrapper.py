
import numpy as np

from gym import ObservationWrapper, spaces


class NormalizeObservationByKey(ObservationWrapper):
    """
    Wrapper to normalize observations.
    Each part of the observation is identified by a key and can be normalized separately
    """

    def __init__(self, env, kv_pairs=dict()):
        super(NormalizeObservationByKey, self).__init__(env)
        self.kv_pairs = kv_pairs

    def observation(self, observation):
        for key in self.kv_pairs.keys():
            if isinstance(observation[key], np.ndarray):
                observation[key] = (observation[key] / self.kv_pairs[key]).astype(np.float32)
            else:
                observation[key] = observation[key] / self.kv_pairs[key]
        return observation


class RGBImgObsRotationWrapper(ObservationWrapper):
    """
    Wrapper to rotate image observation based on agent direction
    """

    def __init__(self, env,):
        super().__init__(env)

    def observation(self, obs):
        env = self.unwrapped
        obs['image'] = np.rot90(obs['image'], k=env.agent_dir)
        return obs


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.

    adapted from gym_minigrid/wrappers.py
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.height*tile_size, self.env.width*tile_size, 3),
            dtype=np.float32
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            tile_size=self.tile_size
        )

        obs['image'] = rgb_img
        return obs



