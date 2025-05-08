"""
Record video of a trained agent's performance in the environments.

This script loads a trained model and records its behavior in the specified environment,
saving the resulting video to a file.
"""

import os
import argparse
import numpy as np
import imageio
import gymnasium as gym
# Import the register module to make environments available
import envs_patch.register_v1_envs
from stable_baselines3 import PPO, DDPG

def record_video(model_path, env_id, output_file, num_episodes=3, seed=None):
    """
    Record a video of a trained agent's performance.
    
    Args:
        model_path (str): Path to the saved model
        env_id (str): Environment ID
        output_file (str): Path to save the video
        num_episodes (int): Number of episodes to record
        seed (int, optional): Random seed for reproducibility
    """
    print(f"Loading model from {model_path}")
    
    # Determine model type from filename
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
        print("Loaded PPO model")
    elif "ddpg" in model_path.lower():
        model = DDPG.load(model_path)
        print("Loaded DDPG model")
    else:
        raise ValueError(f"Unknown model type in {model_path}. Filename should contain 'ppo' or 'ddpg'.")
    
    # Create environment with render_mode set to rgb_array if supported
    try:
        # First try with render_mode parameter
        env = gym.make(env_id, render_mode="rgb_array")
        supports_render_mode = True
    except TypeError:
        # If that fails, try without render_mode
        print("Environment doesn't support render_mode parameter, using default rendering.")
        env = gym.make(env_id)
        supports_render_mode = False
    
    print(f"Created environment: {env_id}")
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Set up manual rendering if environment doesn't support render_mode
    if not supports_render_mode:
        try:
            # Try importing OpenCV for manual rendering
            import cv2
            has_cv2 = True
            
            # Create a simple rendering function
            def create_manual_frame(env, info):
                # Create a blank image
                img_size = 800
                img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
                
                # Draw environment state if we have position information
                if 'aircraft_position' in info:
                    # Scale coordinates to fit in the image
                    pos = info['aircraft_position']
                    center_x, center_y = img_size // 2, img_size // 2
                    scale = img_size / 200.0  # Adjust based on your environment scale
                    
                    # Draw aircraft as blue circle
                    x = int(center_x + pos[0] * scale)
                    y = int(center_y - pos[1] * scale)  # Invert Y for image coordinates
                    if 0 <= x < img_size and 0 <= y < img_size:
                        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
                    
                    # Add text information
                    cv2.putText(img, f"Step: {info.get('step', 'N/A')}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(img, f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if 'aircraft_heading' in info:
                        cv2.putText(img, f"Heading: {info['aircraft_heading']:.1f}°", 
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if 'aircraft_speed' in info:
                        cv2.putText(img, f"Speed: {info['aircraft_speed']:.1f} kts", 
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if 'aircraft_vs' in info:
                        cv2.putText(img, f"VS: {info['aircraft_vs']:.1f} ft/min", 
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                return img
        except ImportError:
            has_cv2 = False
            print("OpenCV not available for manual rendering. Install with 'pip install opencv-python'")
    else:
        has_cv2 = False  # Not needed if environment handles rendering
    
    # Collect frames from multiple episodes
    all_frames = []
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = truncated = False
        episode_frames = []
        episode_reward = 0
        step_count = 0
        
        # Run episode
        while not (done or truncated):
            # Get frame from environment if it supports rendering
            if supports_render_mode:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action
            obs, reward, done, truncated, info = env.step(action)
            
            # Create manual frame if needed
            if not supports_render_mode and has_cv2:
                frame = create_manual_frame(env, info)
                episode_frames.append(frame)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {reward:.2f}, Total reward: {episode_reward:.2f}")
                
                # Debug information
                if 'aircraft_position' in info:
                    print(f"  Position: {info['aircraft_position']}")
                if 'goal_distance' in info:
                    print(f"  Goal distance: {info['goal_distance']:.2f}")
                if 'min_horizontal_separation' in info:
                    print(f"  Min separation: H={info['min_horizontal_separation']:.2f} NM, " +
                          f"V={info.get('min_vertical_separation', 'Unknown'):.0f} ft")
        
        print(f"Episode {episode+1} complete. Reward: {episode_reward:.2f}, Steps: {step_count}")
        total_rewards.append(episode_reward)
        
        # Add episode frames to full video
        all_frames.extend(episode_frames)
    
    # Close environment
    env.close()
    
    # Check if we have frames to save
    if not all_frames:
        print("Warning: No frames were captured. The environment may not support rendering or OpenCV is not installed.")
        return
    
    # Print summary
    print(f"\nRecording summary:")
    print(f"Total frames: {len(all_frames)}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    
    # Save video
    print(f"Saving video to {output_file}")
    imageio.mimsave(output_file, all_frames, fps=15)
    print("Video saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video of a trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--env", type=str, required=True, 
                       help="Environment ID (e.g., 'StaticObstacleEnv-v1', 'SectorCREnv-v1')")
    parser.add_argument("--out", type=str, default="videos/agent_demo.mp4", help="Output video file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    record_video(
        model_path=args.model,
        env_id=args.env,
        output_file=args.out,
        num_episodes=args.episodes,
        seed=args.seed
    )