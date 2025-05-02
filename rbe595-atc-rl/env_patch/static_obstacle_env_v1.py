from typing import Tuple
import numpy as np
import gymnasium as gym
from bluesky_gym.envs.static_obstacle_env import StaticObstacleEnv
from gymnasium.spaces import Box

VS_MIN_FPM, VS_MAX_FPM = -2000.0, 2000.0      # ± 20 ft/s

class StaticObstacleEnvV1(StaticObstacleEnv):
    """
    Adds vertical-speed control and altitude-delta observation.
    Action:  [delta_heading (deg), delta_speed (kts), vertical_speed (ft/min)]
    Obs:      original_obs + [own_alt_ft - obs_alt_ft] for every obstacle
    """
    metadata = StaticObstacleEnv.metadata | {"render_fps": 15}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1) ACTION SPACE  ───────────────────────────────────────────
        # v0: 2-D Box(low=[-15,-30], high=[+15,+30])
        low  = np.append(self.action_space.low,  VS_MIN_FPM)
        high = np.append(self.action_space.high, VS_MAX_FPM)
        self.action_space = Box(low, high, dtype=np.float32)

        # 2) OBS SPACE  ──────────────────────────────────────────────
        extra = 1 * self.num_obstacles                 # one Δ-alt per obstacle
        low  = np.append(self.observation_space.low,  [-1e5]*extra)
        high = np.append(self.observation_space.high, [ 1e5]*extra)
        self.observation_space = Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Return extended observation with altitude deltas."""
        base = super()._get_obs()
        own_alt = self.bluesky.aircraft.get_states()[self.agent_callsign]["altitude"]
        deltas = []
        for obs in self.obstacles:
            deltas.append(own_alt - obs["altitude"])
        return np.concatenate([base, np.array(deltas, dtype=np.float32)])

    # ------------------------------------------------------------------
    def _apply_action(self, action: np.ndarray):
        """Override to forward v-speed to the Bluesky command API."""
        # original two commands
        super()._apply_action(action[:2])

        # NEW: vertical speed command (BlueSky uses ft/min)
        vs_fpm = float(action[2])
        self.bluesky.command(f"{self.agent_callsign} vs {vs_fpm}")