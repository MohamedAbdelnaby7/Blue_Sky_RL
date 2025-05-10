"""
Fixed DDPG script compatible with older versions of Stable-Baselines3.
Addresses the negative actor loss and large positive critic loss issues.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import envs_patch.register_v1_envs
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from gymnasium import spaces

# Simple callback to monitor and log loss values
class LossMonitorCallback(BaseCallback):
    """
    Callback for monitoring actor and critic losses.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.actor_losses = []
        self.critic_losses = []
        self.iteration = 0
    
    def _on_step(self):
        """
        Check losses from logger and print statistics
        """
        # Extract recent losses from logger
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            if "train/actor_loss" in self.model.logger.name_to_value:
                actor_loss = self.model.logger.name_to_value["train/actor_loss"]
                self.actor_losses.append(actor_loss)
            
            if "train/critic_loss" in self.model.logger.name_to_value:
                critic_loss = self.model.logger.name_to_value["train/critic_loss"]
                self.critic_losses.append(critic_loss)
            
            # Log every 10 episodes or when losses are extreme
            self.iteration += 1
            should_log = (self.iteration % 10 == 0)
            
            # Also log if critic loss is very high
            if len(self.critic_losses) > 0:
                last_critic_loss = self.critic_losses[-1]
                if last_critic_loss > 1000:
                    should_log = True
            
            if should_log and (len(self.actor_losses) > 0 or len(self.critic_losses) > 0):
                print("\n--- Loss Statistics ---")
                
                if len(self.actor_losses) > 0:
                    recent_actor_losses = self.actor_losses[-10:]
                    print(f"Actor loss:  {np.mean(recent_actor_losses):.4f} (min: {np.min(recent_actor_losses):.4f}, max: {np.max(recent_actor_losses):.4f})")
                
                if len(self.critic_losses) > 0:
                    recent_critic_losses = self.critic_losses[-10:]
                    print(f"Critic loss: {np.mean(recent_critic_losses):.4f} (min: {np.min(recent_critic_losses):.4f}, max: {np.max(recent_critic_losses):.4f})")
                
                print("----------------------\n")
        
        return True

# Custom callback to manually adjust learning rates for actor and critic
class CustomLRCallback(BaseCallback):
    """
    Callback for manually controlling actor and critic learning rates.
    This works with older versions of Stable-Baselines3 that don't support
    different learning rates for actor and critic.
    """
    def __init__(self, actor_lr=1e-4, critic_lr=5e-5, verbose=0):
        super().__init__(verbose)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.initialized = False
    
    def _on_training_start(self) -> None:
        """
        Called at the start of training, set initial learning rates
        """
        # Set actor and critic learning rates
        if hasattr(self.model, "actor") and hasattr(self.model.actor, "optimizer"):
            self.model.actor.optimizer.param_groups[0]["lr"] = self.actor_lr
            print(f"Set actor learning rate to {self.actor_lr}")
        
        if hasattr(self.model, "critic") and hasattr(self.model.critic, "optimizer"):
            self.model.critic.optimizer.param_groups[0]["lr"] = self.critic_lr
            print(f"Set critic learning rate to {self.critic_lr}")
        
        self.initialized = True
    
    def _on_step(self):
        """
        Called at each step to ensure learning rates remain set correctly
        """
        # Ensure learning rates stay at desired values
        if self.initialized:
            if hasattr(self.model, "actor") and hasattr(self.model.actor, "optimizer"):
                current_actor_lr = self.model.actor.optimizer.param_groups[0]["lr"]
                if current_actor_lr != self.actor_lr:
                    self.model.actor.optimizer.param_groups[0]["lr"] = self.actor_lr
            
            if hasattr(self.model, "critic") and hasattr(self.model.critic, "optimizer"):
                current_critic_lr = self.model.critic.optimizer.param_groups[0]["lr"]
                if current_critic_lr != self.critic_lr:
                    self.model.critic.optimizer.param_groups[0]["lr"] = self.critic_lr
        
        return True

def train_ddpg(env_id: str, 
               total_timesteps: int = 300_000, 
               seed: int = None,
               eval_freq: int = 10_000,
               save_freq: int = 50_000,
               noise_type: str = "ou_noise",    # Use OU noise instead of normal
               noise_scale: float = 0.2,        # Reduced from 0.3
               actor_lr: float = 5e-5,          # Lower actor learning rate
               critic_lr: float = 1e-4,         # Higher critic learning rate
               batch_size: int = 128,           # Increased from 64
               buffer_size: int = 200_000,      # Increased from 100,000
               gradient_steps: int = 2,         # Increased from 1
               gamma: float = 0.995,            # Increased from 0.99 for longer-term rewards
               learning_starts: int = 20000,    # Increased from 10000
               tau: float = 0.005,              # Slower target updates (was 0.01)
               tensorboard_log: str = "tb/ddpg_improved",
               save_path: str = "models",
               verbose: int = 1):
    """
    Train a DDPG agent with fixed critic learning.
    
    Args:
        env_id (str): Environment ID (e.g., 'StaticObstacleEnv-v1' or 'SectorCREnv-v1')
        total_timesteps (int): Total number of timesteps to train for
        seed (int, optional): Random seed for reproducibility
        eval_freq (int): Frequency of evaluation during training (in timesteps)
        save_freq (int): Frequency of model checkpoints (in timesteps)
        noise_scale (float): Scale of exploration noise
        actor_lr (float): Learning rate for actor
        critic_lr (float): Learning rate for critic (lower than actor)
        batch_size (int): Batch size for optimization
        buffer_size (int): Size of the replay buffer
        gradient_steps (int): Number of gradient steps per environment step
        gamma (float): Discount factor
        tensorboard_log (str): Directory for tensorboard logs
        save_path (str): Directory to save models
        verbose (int): Verbosity level (0: no output, 1: info, 2: debug)
    
    Returns:
        The trained DDPG model
    """
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Training DDPG on {env_id} for {total_timesteps} timesteps")
    
    # Create the environment
    def make_env():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    # Wrap in a dummy vec env for normalization and monitoring
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    # Apply VecNormalize with conservative clipping
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,  
        clip_reward=5.0,  # More conservative reward clipping
        gamma=gamma
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        gamma=gamma
    )
    eval_env.obs_rms = env.obs_rms
    
    # Get info about observation space
    test_env = gym.make(env_id)
    
    # Select the appropriate policy
    if isinstance(test_env.observation_space, spaces.Dict):
        policy = "MultiInputPolicy"
        print(f"Using MultiInputPolicy for dictionary observation space")
    else:
        policy = "MlpPolicy"
        print(f"Using MlpPolicy for standard observation space")
    
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
    
    # Add loss monitor callback
    loss_monitor = LossMonitorCallback(verbose=1)
    
    # Add custom learning rate callback
    lr_callback = CustomLRCallback(actor_lr=actor_lr, critic_lr=critic_lr, verbose=1)
    
    # Set up the action noise - using Normal noise for compatibility
    n_actions = test_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_scale * np.ones(n_actions)
    )
    
    # Using a value in between actor_lr and critic_lr for the init
    # The custom callback will override this with separate rates
    avg_lr = (actor_lr + critic_lr) / 2
    
    # Improved policy network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor network
            qf=[256, 256]   # Critic network
        )
    )
    
    # Create the DDPG model with improved parameters
    model = DDPG(
        policy,
        env,
        action_noise=action_noise,
        learning_rate=avg_lr,         # Will be overridden by callback
        buffer_size=buffer_size,      
        learning_starts=10000,        # More steps before starting to learn
        batch_size=batch_size,         
        tau=0.01,                     # Faster target network updates for stability
        gamma=gamma,                  
        train_freq=1,                 # Update every step
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Starting DDPG training on {env_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Noise scale: {noise_scale}")
    print(f"Actor learning rate: {actor_lr}")
    print(f"Critic learning rate: {critic_lr}")
    print(f"Using custom callback to set different learning rates")
    print(f"Batch size: {batch_size}, Buffer size: {buffer_size}")
    print(f"Gradient steps per env step: {gradient_steps}")
    print(f"Target network update rate (tau): 0.01")
    print(f"{'='*50}\n")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, loss_monitor, lr_callback],
        log_interval=1  # Log every episode for better monitoring
    )
    
    # Save the final model
    final_model_path = f"{save_path}/{env_id}-ddpg-fixed"
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    
    print(f"Training complete. Final model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPG with separate learning rates")
    parser.add_argument("--env", type=str, default="StaticObstacleEnv-v1", 
                        help="Environment ID: 'StaticObstacleEnv-v1' or 'SectorCREnv-v1'")
    parser.add_argument("--steps", type=int, default=300_000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    parser.add_argument("--eval-freq", type=int, default=10_000, 
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--save-freq", type=int, default=50_000, 
                        help="Model checkpoint frequency in timesteps")
    parser.add_argument("--noise-scale", type=float, default=0.3, 
                        help="Scale of exploration noise")
    parser.add_argument("--actor-lr", type=float, default=1e-4, 
                        help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=5e-5, 
                        help="Critic learning rate (lower for stability)")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for optimization")
    parser.add_argument("--buffer-size", type=int, default=100_000, 
                        help="Size of the replay buffer")
    parser.add_argument("--gradient-steps", type=int, default=1, 
                        help="Number of gradient steps per environment step")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--tb-log", type=str, default="tb/ddpg_fixed", 
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
        noise_scale=args.noise_scale,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tensorboard_log=args.tb_log,
        save_path=args.save_path,
        verbose=args.verbose
    )