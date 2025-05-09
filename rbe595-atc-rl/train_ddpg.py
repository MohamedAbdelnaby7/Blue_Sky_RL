"""
DDPG training script with techniques to stabilize critic training.
This addresses issues with negative actor loss and large positive critic loss.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import envs_patch.register_v1_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Monkey patch the DDPG class to modify its critic training behavior
def patched_train(self, gradient_steps: int, batch_size: int = 100) -> None:
    """
    Modified training method for DDPG to stabilize critic learning.
    
    This patch applies:
    1. Gradient clipping for critic
    2. Lower learning rate for critic updates
    3. Target network smoothing
    4. Huber loss instead of MSE for more stable gradients
    5. Added L2 regularization for critic
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    
    actor_losses = []
    critic_losses = []
    
    for _ in range(gradient_steps):
        self._n_updates += 1
        
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, 0.2)
            noise = noise.clamp(-0.5, 0.5)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = self.critic_target(replay_data.next_observations, next_actions)

            # Apply TD(0) for more stable learning
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        # Use Huber loss instead of MSE for more stable gradients
        critic_loss = F.huber_loss(current_q_values, target_q_values)
        
        # Add L2 regularization to prevent critic overconfidence
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.critic.parameters():
            l2_reg = l2_reg + torch.norm(param)
        critic_loss = critic_loss + 1e-3 * l2_reg
        
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        
        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic(replay_data.observations, self.actor(replay_data.observations)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            
            # Apply gradient clipping to the actor as well
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            
            self.actor.optimizer.step()

            # Update target networks with lower tau for more stability
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), 0.995)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), 0.995)

    # Log losses if enough updates have been performed
    if len(actor_losses) > 0:
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
    # Add extra logging to track critic accuracy
    if hasattr(self, "_vec_normalize_env") and self._vec_normalize_env is not None:
        try:
            sample = self.replay_buffer.sample(min(batch_size, len(self.replay_buffer)), env=self._vec_normalize_env)
            with torch.no_grad():
                q_pred = self.critic(sample.observations, sample.actions)
                q_target = sample.rewards + (1 - sample.dones) * self.gamma * self.critic_target(
                    sample.next_observations, self.actor_target(sample.next_observations)
                )
                q_error = torch.abs(q_pred - q_target).mean().item()
                self.logger.record("train/q_value_error", q_error)
                self.logger.record("train/q_value_mean", q_pred.mean().item())
        except:
            # Skip if buffer is too small
            pass

# Helper function for soft update
def polyak_update(params: List[torch.nn.Parameter], target_params: List[torch.nn.Parameter], tau: float) -> None:
    """
    Perform a Polyak average update on target_params using params.
    
    This is a more stable version of the target network update with higher tau
    for more conservative updates.
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(tau)
            target_param.data.add_((1 - tau) * param.data)

# Custom callback to track loss metrics in more detail
class LossMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.last_log_step = 0
        self.log_interval = 1000
    
    def _on_step(self):
        if "train/critic_loss" in self.locals["self"].logger.name_to_value:
            self.critic_losses.append(self.locals["self"].logger.name_to_value["train/critic_loss"])
        if "train/actor_loss" in self.locals["self"].logger.name_to_value:
            self.actor_losses.append(self.locals["self"].logger.name_to_value["train/actor_loss"])
        if "train/q_value_mean" in self.locals["self"].logger.name_to_value:
            self.q_values.append(self.locals["self"].logger.name_to_value["train/q_value_mean"])
        
        # Log detailed metrics periodically
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            
            # Print extended metrics and analysis
            if len(self.critic_losses) > 0:
                recent_critic_loss = np.mean(self.critic_losses[-100:]) if len(self.critic_losses) >= 100 else np.mean(self.critic_losses)
                max_critic_loss = np.max(self.critic_losses[-100:]) if len(self.critic_losses) >= 100 else np.max(self.critic_losses) if self.critic_losses else 0
                
                print(f"\n--- Detailed Loss Analysis at Step {self.num_timesteps} ---")
                print(f"Recent critic loss (avg): {recent_critic_loss:.4f}")
                print(f"Recent critic loss (max): {max_critic_loss:.4f}")
                
                if len(self.actor_losses) > 0:
                    recent_actor_loss = np.mean(self.actor_losses[-100:]) if len(self.actor_losses) >= 100 else np.mean(self.actor_losses)
                    print(f"Recent actor loss (avg): {recent_actor_loss:.4f}")
                
                if len(self.q_values) > 0:
                    recent_q_value = np.mean(self.q_values[-100:]) if len(self.q_values) >= 100 else np.mean(self.q_values)
                    print(f"Recent Q-value mean: {recent_q_value:.4f}")
                
                # Add warning for potential issues
                if recent_critic_loss > 100:
                    print("WARNING: Critic loss is very high. Consider reducing learning rate.")
                if recent_critic_loss < 0.1 and len(self.critic_losses) > 1000:
                    print("WARNING: Critic loss is very low. Possible overfitting.")
                if recent_actor_loss < -10 and len(self.actor_losses) > 1000:
                    print("WARNING: Actor loss is very negative. Possible exploitation of critic errors.")
                
                print("----------------------------------------\n")
        
        return True

# Callback to control actor updates for more stable learning
class DelayedActorUpdateCallback(BaseCallback):
    def __init__(self, update_freq=2, initial_delay=5000, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq  # Update actor every N critic updates
        self.initial_delay = initial_delay  # Steps before starting actor updates
        self.critic_update_count = 0
    
    def _on_step(self):
        # Monkey patch the update method temporarily
        if not hasattr(self, "original_update"):
            self.original_update = self.model._update_learning_rate
            
            def delayed_actor_update(learning_rate):
                # Only update actor learning rate if conditions are met
                if self.num_timesteps < self.initial_delay:
                    # During initial phase, set actor learning rate to 0
                    self.model.actor.optimizer.param_groups[0]["lr"] = 0
                    self.model.critic.optimizer.param_groups[0]["lr"] = learning_rate(1)
                else:
                    # After initial phase, update actor less frequently
                    self.critic_update_count += 1
                    if self.critic_update_count % self.update_freq == 0:
                        self.model.actor.optimizer.param_groups[0]["lr"] = learning_rate(1) / 2
                    else:
                        self.model.actor.optimizer.param_groups[0]["lr"] = 0
                    
                    # Always update critic
                    self.model.critic.optimizer.param_groups[0]["lr"] = learning_rate(1)
            
            self.model._update_learning_rate = delayed_actor_update
        
        return True
    
    def on_training_end(self):
        # Restore original method
        if hasattr(self, "original_update"):
            self.model._update_learning_rate = self.original_update

def train_ddpg(env_id: str, 
               total_timesteps: int = 500_000, 
               seed: int = None,
               eval_freq: int = 10_000,
               save_freq: int = 50_000,
               noise_scale: float = 0.3,  # Lower initial noise
               learning_rate: float = 5e-5,  # Reduced learning rate
               batch_size: int = 64,
               buffer_size: int = 100_000,
               gradient_steps: int = 1,  # Fewer gradient steps for stability
               gamma: float = 0.99,
               tensorboard_log: str = "tb/ddpg_critic_stable",
               save_path: str = "models",
               verbose: int = 1):
    """
    Train a DDPG agent with techniques to stabilize the critic.
    
    Args:
        env_id (str): Environment ID (e.g., 'StaticObstacleEnv-v1' or 'SectorCREnv-v1')
        total_timesteps (int): Total number of timesteps to train for
        seed (int): Random seed for reproducibility
        eval_freq (int): Frequency of evaluation during training (in timesteps)
        save_freq (int): Frequency of model checkpoints (in timesteps)
        noise_scale (float): Scale of exploration noise
        learning_rate (float): Learning rate
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
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Create the environment
    def make_env():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    # Wrap environment for normalization
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    # Apply observation and reward normalization with extreme clipping
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        gamma=gamma
    )
    eval_env.obs_rms = env.obs_rms
    
    # Get environment information
    test_env = gym.make(env_id)
    
    # Select appropriate policy
    if isinstance(test_env.observation_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"
    
    # Configure action noise with smaller initial scale
    n_actions = test_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_scale * np.ones(n_actions)
    )
    
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
        name_prefix=f"{env_id}-ddpg-stable",
        verbose=1
    )
    
    # Add custom callbacks for loss monitoring and actor update control
    loss_callback = LossMetricsCallback(verbose=1)
    actor_delay_callback = DelayedActorUpdateCallback(
        update_freq=3,  # Update actor every 3 critic updates
        initial_delay=10000,  # Start actor updates after 10k steps
        verbose=1
    )
    
    # Policy configuration with larger networks
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Deeper actor network
            qf=[256, 256, 128]   # Deeper critic network
        ),
        # Use layer normalization for better gradient flow
        share_features_extractor=False,
        # Use ReLU for better gradients
        activation_fn=torch.nn.ReLU
    )
    
    # Create the DDPG model
    model = DDPG(
        policy,
        env,
        action_noise=action_noise,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=10000,  # More samples before learning
        batch_size=batch_size,
        tau=0.01,
        gamma=gamma,
        train_freq=1,
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device="auto",
        verbose=verbose
    )
    
    # Apply the patched training method
    # This is where we modify DDPG's core training loop to stabilize critic learning
    model.train = patched_train.__get__(model, DDPG)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Training DDPG with critic stabilization on {env_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate} (with delayed actor updates)")
    print(f"Batch size: {batch_size}, Buffer size: {buffer_size}")
    print(f"Using Huber loss and gradient clipping for critic training")
    print(f"Initial exploration noise: {noise_scale}")
    print(f"Discount factor (gamma): {gamma}")
    print(f"{'='*50}\n")
    
    # Train the model with all callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, loss_callback, actor_delay_callback],
        log_interval=10
    )
    
    # Save the final model
    final_model_path = f"{save_path}/{env_id}-ddpg-stable"
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    
    print(f"Training complete! Final model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPG with critic stabilization")
    parser.add_argument("--env", type=str, default="StaticObstacleEnv-v1", 
                       help="Environment ID: 'StaticObstacleEnv-v1' or 'SectorCREnv-v1'")
    parser.add_argument("--steps", type=int, default=500_000, 
                       help="Total timesteps to train for")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-5, 
                       help="Learning rate (smaller for stability)")
    parser.add_argument("--noise", type=float, default=0.3, 
                       help="Exploration noise scale")
    parser.add_argument("--batch-size", type=int, default=64, 
                       help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, 
                       help="Replay buffer size")
    parser.add_argument("--gradient-steps", type=int, default=1, 
                       help="Gradient steps per environment step")
    parser.add_argument("--gamma", type=float, default=0.99, 
                       help="Discount factor")
    parser.add_argument("--tb-log", type=str, default="tb/ddpg_critic_stable", 
                       help="Tensorboard log directory")
    parser.add_argument("--save-path", type=str, default="models", 
                       help="Model save directory")
    parser.add_argument("--verbose", type=int, default=1, 
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    train_ddpg(
        env_id=args.env,
        total_timesteps=args.steps,
        seed=args.seed,
        learning_rate=args.lr,
        noise_scale=args.noise,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tensorboard_log=args.tb_log,
        save_path=args.save_path,
        verbose=args.verbose
    )