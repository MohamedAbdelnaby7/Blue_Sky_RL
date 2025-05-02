# envs_patch/sector_cr_env_v1.py
"""
SectorCREnv-v1
==============

Extends the original *SectorCREnv-v0* with:
1. 3-D continuous action vector: [Δ-heading (deg), Δ-speed (kts), vertical_speed (ft/min)]
2. Observation augmented by one altitude-delta (own_alt − intruder_alt) per intruder.
"""

from typing import List
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from bluesky_gym.envs.sector_cr_env import SectorCREnv

# ──────────────────────────────────────────────────────────────────────────────
VS_MIN_FPM, VS_MAX_FPM = -2000.0, 2000.0      # command range ±2 000 ft/min


class SectorCREnvV1(SectorCREnv):
    """Vertical-speed-aware version of Sector Conflict Resolution environment."""

    metadata = SectorCREnv.metadata | {"render_fps": 15}

    # --------------------------------------------------------------------- init
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1) ACTION SPACE  (old: 2-D) ➜  3-D  ──────────────────────────────
        low  = np.append(self.action_space.low,  VS_MIN_FPM)
        high = np.append(self.action_space.high, VS_MAX_FPM)
        self.action_space = Box(low, high, dtype=np.float32)

        # 2) OBSERVATION SPACE (old + Δ-alt per intruder) ─────────────────
        extra = len(self.intruders)            # one value per intruder
        low  = np.append(self.observation_space.low,  [-1e5] * extra)
        high = np.append(self.observation_space.high, [ 1e5] * extra)
        self.observation_space = Box(low, high, dtype=np.float32)

    # ---------------------------------------------------------------- get_obs
    def _get_obs(self) -> np.ndarray:
        """
        Original observation + altitude deltas.
        """
        base_obs = super()._get_obs()

        own_alt = self.bluesky.aircraft.get_states()[self.agent_callsign]["altitude"]

        alt_deltas: List[float] = []
        for intr in self.intruders:
            alt_deltas.append(own_alt - intr["altitude"])

        return np.concatenate([base_obs, np.array(alt_deltas, dtype=np.float32)])

    # ------------------------------------------------------------- apply_action
    def _apply_action(self, action: np.ndarray):
        """
        Forward (Δ-heading, Δ-speed) to the parent implementation,
        then send the new vertical-speed command.
        """
        # First two channels handled by parent class
        super()._apply_action(action[:2])

        # Channel 3: vertical speed in ft/min                     ---------------
        vs_fpm = float(action[2])
        self.bluesky.command(f"{self.agent_callsign} vs {vs_fpm}")