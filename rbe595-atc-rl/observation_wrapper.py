import gymnasium as gym
import numpy as np

class SimpleObsWrapper(gym.Wrapper):
    """Simple wrapper that just clips observations to defined bounds."""
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Keep the original observation space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._clip_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._clip_observation(obs), reward, terminated, truncated, info
    
    def _clip_observation(self, obs):
        """Clip observation to be within observation space."""
        if isinstance(self.observation_space, gym.spaces.Box):
            return np.clip(obs, self.observation_space.low, self.observation_space.high)
        elif isinstance(self.observation_space, gym.spaces.Dict):
            clipped = {}
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    clipped[key] = np.clip(obs[key], space.low, space.high)
                else:
                    clipped[key] = obs[key]
            return clipped
        return obs

# Usage
env = SimpleObsWrapper(gym.make("StaticObstacleEnv-v1"))