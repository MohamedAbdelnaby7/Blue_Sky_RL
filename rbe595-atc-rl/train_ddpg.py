"""
Train a Deep Deterministic Policy Gradient (DDPG) agent on the extended environments.

This script trains DDPG agents for either the StaticObstacleEnv-v1 or SectorCREnv-v1
environment, with vertical speed control capabilities.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import envs_patch.register_v1_envs
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gymnasium import spaces

def train_ddpg(env_id: str, 
               total_timesteps: int = 1_000_000, 
               seed: int = None,
               eval_freq: int = 50_000,
               save_freq: int = 100_000,
               noise_type: str = "normal",
               noise_scale: float = 0.3,
               tensorboard_log: str = "tb/ddpg",
               save_path: str = "models",
               verbose: int = 1):
    """
    Train a DDPG agent on the specified environment.
    
    Args:
        env_id (str): Environment ID (e.g., 'StaticObstacleEnv-v1', 'SectorCREnv-v1')
        total_timesteps (int): Total number of timesteps to train for
        seed (int, optional): Random seed for reproducibility
        eval_freq (int): Frequency of evaluation during training (in timesteps)
        save_freq (int): Frequency of model checkpoints (in timesteps)
        noise_type (str): Type of exploration noise ('normal' or 'ou')
        noise_scale (float): Scale of the exploration noise
        tensorboard_log (str): Directory for tensorboard logs
        save_path (str): Directory to save models
        verbose (int): Verbosity level (0: no output, 1: info, 2: debug)
    
    Returns:
        The trained DDPG model
    """
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Create the environment
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_env.reset(seed=seed)
    
    # Select the appropriate policy based on observation space type
    if isinstance(env.observation_space, spaces.Dict):
        policy = "MultiInputPolicy"
        print(f"Using MultiInputPolicy for dictionary observation space: {env.observation_space}")
    else:
        policy = "MlpPolicy"
        print(f"Using MlpPolicy for non-dictionary observation space: {env.observation_space}")
    
    # Define callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=f"{save_path}/logs",
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{save_path}/checkpoints",
        name_prefix=f"{env_id}-ddpg",
        verbose=1
    )
    
    # Set up the action noise for exploration
    n_actions = env.action_space.shape[0]
    mean = np.zeros(n_actions)
    sigma = noise_scale * np.ones(n_actions)
    
    if noise_type == "normal":
        action_noise = NormalActionNoise(mean=mean, sigma=sigma)
    elif noise_type == "ou":
        action_noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Use 'normal' or 'ou'")
    
    # Set up the DDPG model with optimized hyperparameters
    model = DDPG(
        policy,
        env,
        action_noise=action_noise,
        learning_rate=1e-3,
        buffer_size=1_000_000,   # Size of the replay buffer
        learning_starts=10000,   # Number of steps before learning starts
        batch_size=256,          # Batch size for optimization
        tau=0.005,               # Soft update coefficient
        gamma=0.99,              # Discount factor
        train_freq=(1, "episode"), # Update the model every episode
        gradient_steps=-1,       # Number of gradient steps to take (-1 means same as batch size)
        policy_kwargs=dict(
            net_arch=dict(
                pi=[400, 300],   # Actor network architecture
                qf=[400, 300]    # Critic network architecture
            )
        ),
        tensorboard_log=tensorboard_log,
        verbose=verbose
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10
    )
    
    # Save the final model
    final_model_path = f"{save_path}/{env_id}-ddpg"
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPG agent on ATC environments with vertical speed control")
    parser.add_argument("--env", type=str, default="StaticObstacleEnv-v1", 
                        help="Environment ID: 'StaticObstacleEnv-v1' or 'SectorCREnv-v1'")
    parser.add_argument("--steps", type=int, default=1_000_000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    parser.add_argument("--eval-freq", type=int, default=50_000, 
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--save-freq", type=int, default=100_000, 
                        help="Model checkpoint frequency in timesteps")
    parser.add_argument("--noise", type=str, default="normal", choices=["normal", "ou"],
                        help="Type of exploration noise ('normal' or 'ou')")
    parser.add_argument("--noise-scale", type=float, default=0.3, 
                        help="Scale of the exploration noise")
    parser.add_argument("--tb-log", type=str, default="tb/ddpg", 
                        help="Tensorboard log directory")
    parser.add_argument("--save-path", type=str, default="models", 
                        help="Directory to save models")
    parser.add_argument("--verbose", type=int, default=1, 
                        help="Verbosity level (0: no output, 1: info, 2: debug)")
    
    args = parser.parse_args()
    
    # Train the DDPG agent
    trained_model = train_ddpg(
        env_id=args.env,
        total_timesteps=args.steps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        noise_type=args.noise,
        noise_scale=args.noise_scale,
        tensorboard_log=args.tb_log,
        save_path=args.save_path,
        verbose=args.verbose
    )