from typing import Tuple
import numpy as np
import gymnasium as gym
from bluesky_gym.envs.static_obstacle_env import StaticObstacleEnv
from gymnasium.spaces import Box
from gymnasium import spaces

VS_MIN_FPM, VS_MAX_FPM = -2000.0, 2000.0      # ± 20 ft/s

class StaticObstacleEnvV1(StaticObstacleEnv):
    """
    Adds vertical-speed control and altitude-delta observation.
    Action:  [delta_heading (deg), delta_speed (kts), vertical_speed (ft/min)]
    Obs:      original_obs + [own_alt_ft - obs_alt_ft] for every obstacle
    """
    metadata = StaticObstacleEnv.metadata | {"render_fps": 15}

    def __init__(self):
        super(StaticObstacleEnvV1, self).__init__()

        # Define action space: [heading, speed, vertical speed]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, -10.0]), high=np.array([1.0, 1.0, 10.0]), dtype=np.float32)

        # Define observation space: [altitude difference, relative position, speed, vertical speed]
        self.observation_space = spaces.Dict({
            "altitude_diff": spaces.Box(low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32),
            "relative_position": spaces.Box(low=-1000.0, high=1000.0, shape=(2,), dtype=np.float32),  # [x, y]
            "speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "vertical_speed": spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
        })

        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        # Generate random obstacles at various altitudes
        num_obstacles = 10
        return np.random.uniform(low=-1000.0, high=10000.0, size=(num_obstacles, 3))  # x, y, altitude

    def reset(self, seed=None, options=None):
        # Reset environment state and accept the seed argument
        if seed is not None:
            np.random.seed(seed)  # Set seed for random number generation
            self._seed = seed  # Optionally save the seed for internal use

        self.state = {
            "altitude_diff": np.array([0.0]),
            "relative_position": np.array([0.0, 0.0]),
            "speed": np.array([0.5]),
            "vertical_speed": np.array([0.0]),
        }

        # Return the state and info (return_info flag used here for compatibility with VecEnv)
        return self.state, {}
    
    def step(self, action):
        # Apply action (heading, speed, vertical speed) to aircraft
        heading, speed, vertical_speed = action

        # Simulate new state
        self.state["relative_position"] += speed  # relative position update
        self.state["altitude_diff"] += vertical_speed  # altitude difference update
        self.state["speed"] = np.array([speed])
        self.state["vertical_speed"] = np.array([vertical_speed])

        # Check for collisions with obstacles
        done = self.check_collision()

        # Handle termination and truncation
        truncated = False  # Set to True if you have a condition for truncation
        if done:
            # If the episode is done (due to collision), you can consider it as a terminated episode
            terminated = True
        else:
            terminated = False

        reward = self.compute_reward()

        # Return observation, reward, done, truncated, and info
        info = {}
        return self.state, reward, terminated, truncated, info

    def check_collision(self):
        # Check if the aircraft collides with any obstacles
        for obs in self.obstacles:
            if np.abs(self.state["altitude_diff"] - obs[2]) < 50:  # If altitude difference is too small
                return True
        return False

    def compute_reward(self):
        # Reward function: negative reward for collision, positive for progress
        if self.check_collision():
            return -100
        return 1  # Reward for not colliding

    def render(self):
        # Render the environment state
        print(f"State: {self.state}")