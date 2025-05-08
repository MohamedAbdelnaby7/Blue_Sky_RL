import gymnasium as gym
import numpy as np
import argparse
import envs_patch.register_v1_envs
from stable_baselines3 import PPO, DDPG
import os
import imageio

def evaluate_agent(env_id, model_path, algo="ppo", num_episodes=5, record=False, video_path=None):
    """
    Evaluate a trained agent in the specified environment.
    
    Args:
        env_id: Environment ID (e.g., "StaticObstacleEnv-v1" or "SectorCREnv-v1")
        model_path: Path to the saved model
        algo: Algorithm used for training ("ppo" or "ddpg")
        num_episodes: Number of episodes to evaluate
        record: Whether to record video
        video_path: Path to save the video (if record=True)
    """
    # Create environment without render_mode parameter
    env = gym.make(env_id)
    
    # Load the trained model
    if algo.lower() == "ppo":
        model = PPO.load(model_path, env=env)
    elif algo.lower() == "ddpg":
        model = DDPG.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Use 'ppo' or 'ddpg'.")
    
    # Run evaluation episodes
    total_rewards = []
    episode_lengths = []
    frames = [] if record else None
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated):
            # Try to render if recording (this may not work depending on environment implementation)
            if record:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except:
                    print("Warning: Rendering not supported in this environment")
                    record = False
                    frames = None
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Display step information
            if step_count % 10 == 0:
                print(f"  Step {step_count}, Reward: {reward:.2f}, Total: {episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.2f} in {step_count} steps")
    
    # Summarize results
    print("\nEvaluation Summary:")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
    
    # Save video if recording was successful
    if record and frames and len(frames) > 0 and video_path:
        print(f"Saving video to {video_path}")
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
        imageio.mimsave(video_path, frames, fps=15)
    
    env.close()
    return total_rewards, episode_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent and optionally record video")
    parser.add_argument("--env", type=str, default="StaticObstacleEnv-v1", help="Environment ID")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "ddpg"], help="Algorithm (ppo or ddpg)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--record", action="store_true", help="Record video of the agent")
    parser.add_argument("--out", type=str, help="Path to save the video (required if --record is set)")
    
    args = parser.parse_args()
    
    if args.record and not args.out:
        parser.error("--out is required when --record is set")
    
    evaluate_agent(
        env_id=args.env,
        model_path=args.model,
        algo=args.algo,
        num_episodes=args.episodes,
        record=args.record,
        video_path=args.out
    )