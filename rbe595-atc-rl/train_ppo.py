"""
Train a Proximal Policy Optimization (PPO) agent on the extended environments.

This script trains PPO agents for either the StaticObstacleEnv-v1 or SectorCREnv-v1
environment, with vertical speed control capabilities.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import envs_patch.register_v1_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from gymnasium import spaces

def train_ppo(env_id: str, 
              total_timesteps: int = 1_000_000, 
              seed: int = None,
              eval_freq: int = 50_000,
              save_freq: int = 100_000,
              tensorboard_log: str = "tb/ppo",
              save_path: str = "models",
              verbose: int = 1):
    """
    Train a PPO agent on the specified environment.
    
    Args:
        env_id (str): Environment ID (e.g., 'StaticObstacleEnv-v1', 'SectorCREnv-v1')
        total_timesteps (int): Total number of timesteps to train for
        seed (int, optional): Random seed for reproducibility
        eval_freq (int): Frequency of evaluation during training (in timesteps)
        save_freq (int): Frequency of model checkpoints (in timesteps)
        tensorboard_log (str): Directory for tensorboard logs
        save_path (str): Directory to save models
        verbose (int): Verbosity level (0: no output, 1: info, 2: debug)
    
    Returns:
        The trained PPO model
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
        name_prefix=f"{env_id}-ppo",
        verbose=1
    )
    
    # Set up the PPO model with optimized hyperparameters
    model = PPO(
        policy,
        env,
        learning_rate=3e-4,
        n_steps=2048,        # Collect more steps before each update
        batch_size=64,       # Mini-batch size for gradient updates
        n_epochs=10,         # Number of epochs when optimizing the surrogate loss
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # GAE factor
        clip_range=0.2,      # Clipping parameter for PPO
        ent_coef=0.01,       # Entropy coefficient (encourages exploration)
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Max norm of the gradient (prevents exploding gradients)
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Actor network architecture
                vf=[256, 256]   # Critic network architecture
            )
        )
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = f"{save_path}/{env_id}-ppo"
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on ATC environments with vertical speed control")
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
    parser.add_argument("--tb-log", type=str, default="tb/ppo", 
                        help="Tensorboard log directory")
    parser.add_argument("--save-path", type=str, default="models", 
                        help="Directory to save models")
    parser.add_argument("--verbose", type=int, default=1, 
                        help="Verbosity level (0: no output, 1: info, 2: debug)")
    
    args = parser.parse_args()
    
    # Train the PPO agent
    trained_model = train_ppo(
        env_id=args.env,
        total_timesteps=args.steps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        tensorboard_log=args.tb_log,
        save_path=args.save_path,
        verbose=args.verbose
    )