"""
Call   >>> import envs_patch.register_v1_envs   once at boot and both
`StaticObstacleEnv-v1` and `SectorCREnv-v1` become available to Gymnasium.
"""
import gymnasium as gym
from .static_obstacle_env_v1 import StaticObstacleEnvV1
from .sector_cr_env_v1 import SectorCREnvV1

gym.register(
    id="StaticObstacleEnv-v1",
    entry_point="envs_patch.static_obstacle_env_v1:StaticObstacleEnvV1",
)
gym.register(
    id="SectorCREnv-v1",
    entry_point="envs_patch.sector_cr_env_v1:SectorCREnvV1",
)