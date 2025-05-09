"""
A clean version of the static obstacle environment with proper 3D navigation
and reduced console output for better training summaries.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

# Define vertical speed command limits (in ft/min)
VS_MIN_FPM, VS_MAX_FPM = -2000.0, 2000.0

class StaticObstacleEnvV1(gym.Env):
    """
    3D Air Traffic Control environment with static obstacles.
    
    This environment simulates an aircraft navigating through 3D airspace
    with static obstacles. The agent must control heading, speed, and vertical speed
    to avoid collisions and reach a goal position.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    
    def __init__(self, render_mode=None, verbose=False):
        """Initialize the environment with extended action and observation spaces."""
        super().__init__()
        
        # Store render mode and verbosity level
        self.render_mode = render_mode
        self.verbose = verbose  # Set to False to reduce console output
        
        # Print initialization information only in verbose mode
        if self.verbose:
            print(f"Initializing StaticObstacleEnvV1 from {os.path.basename(__file__)}")
        
        # Environment parameters
        self.num_obstacles = 10
        self.max_steps = 500
        self.current_step = 0
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([-45.0, -10.0, -2000.0]),
            high=np.array([45.0, 10.0, 2000.0]),
            dtype=np.float32
        )
        
        # Define observation space as a flat Box for simplicity and compatibility
        # This includes all the information the agent needs
        self.observation_space = spaces.Box(
            low=np.array([
                -10000.0, -10000.0, 0.0,      # Position min (x, y, z)
                0.0, 0.0,                     # Heading components (cos, sin)
                150.0,                        # Min speed
                -2000.0,                      # Min vertical speed
                -10000.0, -10000.0, -10000.0, # Goal vector min
                0.0                           # Min goal distance
            ]),
            high=np.array([
                10000.0, 10000.0, 30000.0,    # Position max (x, y, z)
                1.0, 1.0,                     # Heading components (cos, sin)
                350.0,                        # Max speed
                2000.0,                       # Max vertical speed
                10000.0, 10000.0, 10000.0,    # Goal vector max
                20000.0                       # Max goal distance
            ]),
            dtype=np.float32
        )
        
        # Print debug info only in verbose mode
        if self.verbose:
            print(f"Action space: {self.action_space}")
            print(f"Observation space: {self.observation_space}")
        
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
        # Print reset info only in verbose mode
        if self.verbose:
            print("\n--- RESETTING ENVIRONMENT ---")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize aircraft state
        self.aircraft = {
            "position": np.array([0.0, 0.0, 10000.0]),  # Initial position (x, y, altitude in ft)
            "heading": 0.0,                             # Initial heading (degrees)
            "speed": 250.0,                             # Initial speed (knots)
            "vertical_speed": 0.0                       # Initial vertical speed (ft/min)
        }
        
        # Set goal position (far ahead with some lateral offset)
        self.goal_position = np.array([8000.0, 8000.0, 10000.0])  # Goal at (x=8000, y=8000, altitude=10000)
        self.initial_goal_distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        self.prev_goal_distance = self.initial_goal_distance  # For progress tracking
        
        # Generate obstacles in the path to goal
        self.obstacles = []
        
        # Place some obstacles along the path from start to goal
        for i in range(5):
            # Position along the path to goal (spread between 20% and 80% of the way)
            t = 0.2 + 0.6 * (i / 4)
            pos = t * self.goal_position + (1 - t) * self.aircraft["position"]
            
            # Add some randomness
            pos += np.random.uniform(-1000.0, 1000.0, 3)
            
            # Ensure reasonable altitude
            pos[2] = np.clip(pos[2], 5000.0, 15000.0)
            
            # Random radius
            radius = np.random.uniform(300.0, 600.0)
            
            self.obstacles.append({
                "position": pos,
                "radius": radius
            })
        
        # Add some obstacles that require vertical avoidance
        for i in range(3):
            # Position along the path to goal
            t = 0.3 + 0.4 * (i / 2)
            pos = t * self.goal_position + (1 - t) * self.aircraft["position"]
            
            # Keep x,y close to path but vary z significantly
            pos[0] += np.random.uniform(-500.0, 500.0)
            pos[1] += np.random.uniform(-500.0, 500.0)
            pos[2] = 10000.0 + np.random.choice([-2000.0, 2000.0])  # Force vertical deviation
            
            # Larger radius for these obstacles
            radius = np.random.uniform(700.0, 1000.0)
            
            self.obstacles.append({
                "position": pos,
                "radius": radius
            })
        
        # Fill in remaining obstacles with random placement
        for i in range(self.num_obstacles - len(self.obstacles)):
            x = np.random.uniform(-2000.0, 10000.0)
            y = np.random.uniform(-2000.0, 10000.0)
            z = np.random.uniform(5000.0, 15000.0)
            radius = np.random.uniform(300.0, 600.0)
            
            self.obstacles.append({
                "position": np.array([x, y, z]),
                "radius": radius
            })
        
        # Track closest approach to goal for reward shaping
        self.closest_to_goal = self.initial_goal_distance
        
        # Print debug info only in verbose mode
        if self.verbose:
            print(f"Aircraft start: pos={self.aircraft['position']}, heading={self.aircraft['heading']:.1f}°")
            print(f"Goal: pos={self.goal_position}, distance={self.initial_goal_distance:.1f} ft")
            print(f"Placed {len(self.obstacles)} obstacles")
            print("Environment reset complete.")
        
        # Generate initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """
        Apply action and advance the simulation.
        
        Args:
            action: [Δ-heading (deg), Δ-speed (kts), vertical_speed (ft/min)]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Increment step counter
        self.current_step += 1
        
        # Unpack and clip action to valid ranges
        delta_heading = np.clip(float(action[0]), -45.0, 45.0)
        delta_speed = np.clip(float(action[1]), -10.0, 10.0)
        target_vs = np.clip(float(action[2]), -2000.0, 2000.0)
        
        # Store previous position for progress calculation
        prev_position = self.aircraft["position"].copy()
        prev_goal_distance = np.linalg.norm(self.goal_position - prev_position)
        
        # Update aircraft state
        # 1. Update heading (in degrees, keep in [0, 360])
        new_heading = (self.aircraft["heading"] + delta_heading) % 360.0
        self.aircraft["heading"] = new_heading
        
        # 2. Update speed (clip to [150, 350] knots)
        new_speed = np.clip(self.aircraft["speed"] + delta_speed, 150.0, 350.0)
        self.aircraft["speed"] = new_speed
        
        # 3. Set vertical speed
        self.aircraft["vertical_speed"] = target_vs
        
        # 4. Update position based on heading, speed and vertical speed
        # Convert heading to radians for movement calculation
        heading_rad = np.radians(new_heading)
        
        # Convert speed from knots to ft/sec (1 knot ≈ 1.69 ft/sec)
        speed_ftps = new_speed * 1.68781
        
        # Calculate horizontal movement (assuming 1-second time step)
        dx = speed_ftps * np.sin(heading_rad)
        dy = speed_ftps * np.cos(heading_rad)
        
        # Calculate vertical movement (ft/min to ft/sec)
        dz = target_vs / 60.0
        
        # Update position
        self.aircraft["position"][0] += dx
        self.aircraft["position"][1] += dy
        self.aircraft["position"][2] += dz
        
        # Constrain altitude to be > 1000 ft and < 30000 ft
        self.aircraft["position"][2] = np.clip(self.aircraft["position"][2], 1000.0, 30000.0)
        
        # Check for collision with obstacles
        collision = self._check_collision()
        
        # Check if goal reached
        goal_distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        self.closest_to_goal = min(self.closest_to_goal, goal_distance)
        goal_reached = goal_distance < 500.0  # Goal reached if within 500 ft
        
        # Check if out of bounds (far from goal region)
        out_of_bounds = (
            abs(self.aircraft["position"][0]) > 15000.0 or
            abs(self.aircraft["position"][1]) > 15000.0
        )
        
        # Only print periodic status updates in verbose mode
        if self.verbose and self.current_step % 50 == 0:
            print(f"\n-- Step {self.current_step} --")
            print(f"Position: {self.aircraft['position']}")
            print(f"Goal distance: {goal_distance:.1f} (closest so far: {self.closest_to_goal:.1f})")
        
        # Determine outcome and reward
        if collision:
            reward = -100.0
            terminated = True
            if self.verbose:
                print("Collision with obstacle!")
        elif goal_reached:
            reward = 100.0 + (self.max_steps - self.current_step) * 0.1  # Bonus for reaching goal quickly
            terminated = True
            if self.verbose:
                print("Goal reached!")
        elif out_of_bounds:
            reward = -50.0
            terminated = True
            if self.verbose:
                print("Aircraft out of bounds!")
        else:
            # Calculate reward
            reward = self._compute_reward(prev_position, prev_goal_distance)
            terminated = False
        
        # Check if episode should end due to step limit
        truncated = self.current_step >= self.max_steps
        if truncated and self.verbose:
            print("Episode truncated at max steps")
        
        # Get observation for next step
        observation = self._get_obs()
        
        # Create info dict
        info = self._get_info()
        info.update({
            "collision": collision,
            "goal_reached": goal_reached,
            "out_of_bounds": out_of_bounds,
        })
        
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, prev_position, prev_goal_distance):
        """
        Compute reward based on current state and progress.
        
        Args:
            prev_position: Previous aircraft position
            prev_goal_distance: Previous distance to goal
            
        Returns:
            float: Reward value
        """
        # 1. Base reward (small penalty for each step to encourage efficiency)
        reward = -0.1
        
        # 2. Goal-directed reward components
        current_goal_distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        
        # Distance progress reward (reward for getting closer to goal)
        distance_progress = prev_goal_distance - current_goal_distance
        progress_reward = distance_progress * 5.0  # Scale factor - significant reward for progress
        
        # Distance-based reward (higher reward when closer to goal)
        goal_proximity_reward = 10.0 / (1.0 + 0.0001 * current_goal_distance)
        
        # New best approach reward (small bonus for reaching a new closest point to goal)
        if current_goal_distance < self.closest_to_goal + 1.0:  # Adding small margin
            best_approach_reward = 1.0
        else:
            best_approach_reward = 0.0
        
        # 3. Safety rewards
        safety_reward = 0.0
        
        # Penalty for getting too close to obstacles
        for obs in self.obstacles:
            distance = np.linalg.norm(self.aircraft["position"] - obs["position"])
            
            # If within 2x radius, add penalty based on proximity
            if distance < obs["radius"] * 2.0:
                margin = (distance - obs["radius"]) / obs["radius"]  # 0 at surface, 1 at 2x radius
                safety_reward -= (1.0 - margin) * 2.0  # Stronger penalty as we get closer
        
        # 4. Efficiency rewards
        
        # Penalize excessive altitude changes to encourage smooth flight
        vs_penalty = -0.01 * abs(self.aircraft["vertical_speed"]) / 1000.0
        
        # Penalize very slow speeds
        speed_reward = 0.01 * (self.aircraft["speed"] - 150.0) / 200.0  # Max at 350 knots
        
        # 5. Combine all rewards
        total_reward = (
            reward + 
            progress_reward + 
            goal_proximity_reward + 
            best_approach_reward +
            safety_reward + 
            vs_penalty +
            speed_reward
        )
        
        # Print reward breakdown only in verbose mode and only occasionally
        if self.verbose and self.current_step % 50 == 0:
            print(f"Reward: {total_reward:.2f} = Base: -0.1 + Progress: {progress_reward:.2f} + Proximity: {goal_proximity_reward:.2f}")
            print(f"  + Best: {best_approach_reward:.2f} + Safety: {safety_reward:.2f} + VS: {vs_penalty:.2f} + Speed: {speed_reward:.2f}")
        
        return total_reward

    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
            ndarray: Observation vector
        """
        # Aircraft position
        pos_x, pos_y, pos_z = self.aircraft["position"]
        
        # Convert heading to sine and cosine components for continuity
        heading_rad = np.radians(self.aircraft["heading"])
        heading_cos = np.cos(heading_rad)
        heading_sin = np.sin(heading_rad)
        
        # Aircraft speed and vertical speed
        speed = self.aircraft["speed"]
        vspeed = self.aircraft["vertical_speed"]
        
        # Goal information
        goal_vector = self.goal_position - self.aircraft["position"]
        goal_distance = np.linalg.norm(goal_vector)
        
        # Combine all information into a flat observation vector
        obs = np.array([
            pos_x, pos_y, pos_z,                # Aircraft position
            heading_cos, heading_sin,           # Heading components
            speed,                              # Aircraft speed
            vspeed,                             # Vertical speed
            goal_vector[0], goal_vector[1], goal_vector[2],  # Vector to goal
            goal_distance                       # Distance to goal
        ], dtype=np.float32)
        
        return obs

    def _get_info(self):
        """
        Get additional information about the environment state.
        
        Returns:
            dict: Additional information
        """
        # Calculate distance to goal
        goal_distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        
        # Find closest obstacle and distance to it
        closest_distance = float('inf')
        for obs in self.obstacles:
            distance = np.linalg.norm(self.aircraft["position"] - obs["position"]) - obs["radius"]
            closest_distance = min(closest_distance, distance)
        
        return {
            "aircraft_position": self.aircraft["position"].copy(),
            "aircraft_heading": float(self.aircraft["heading"]),
            "aircraft_speed": float(self.aircraft["speed"]),
            "aircraft_vs": float(self.aircraft["vertical_speed"]),
            "goal_distance": goal_distance,
            "closest_obstacle_distance": closest_distance,
            "closest_approach_to_goal": self.closest_to_goal,
            "step": self.current_step
        }

    def _check_collision(self):
        """
        Check if aircraft has collided with any obstacle.
        
        Returns:
            bool: True if collision occurred, False otherwise
        """
        for obs in self.obstacles:
            # Calculate 3D distance to obstacle center
            distance = np.linalg.norm(self.aircraft["position"] - obs["position"])
            
            # Check if distance is less than obstacle radius
            if distance < obs["radius"]:
                return True
                
        return False

    def render(self):
        """
        Render the environment.
        
        Returns:
            ndarray or None: If render_mode is 'rgb_array', returns an RGB image,
                             otherwise returns None
        """
        if self.render_mode is None:
            return None
            
        # For 'rgb_array' mode, create a visualization
        if self.render_mode == "rgb_array":
            try:
                import cv2
                has_cv2 = True
            except ImportError:
                has_cv2 = False
                if self.verbose:
                    print("OpenCV not available, using simple rendering.")
            
            # Create a top-down view image
            img_size = 800
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Define coordinate transformation functions
            world_min_x, world_max_x = -3000, 11000
            world_min_y, world_max_y = -3000, 11000
            world_width = world_max_x - world_min_x
            world_height = world_max_y - world_min_y
            
            def world_to_img_x(x):
                return int((x - world_min_x) / world_width * img_size)
                
            def world_to_img_y(y):
                # Flip Y axis for image coordinates
                return int(img_size - (y - world_min_y) / world_height * img_size)
            
            if has_cv2:
                # Draw coordinate grid (every 1000 ft)
                for x in range(int(world_min_x/1000)*1000, int(world_max_x/1000+1)*1000, 1000):
                    cv2.line(img, 
                        (world_to_img_x(x), 0), 
                        (world_to_img_x(x), img_size), 
                        (220, 220, 220), 1)
                    if x % 5000 == 0:  # Label every 5000 ft
                        cv2.putText(img, f"{x}", 
                            (world_to_img_x(x)+5, img_size-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                for y in range(int(world_min_y/1000)*1000, int(world_max_y/1000+1)*1000, 1000):
                    cv2.line(img, 
                        (0, world_to_img_y(y)), 
                        (img_size, world_to_img_y(y)), 
                        (220, 220, 220), 1)
                    if y % 5000 == 0:  # Label every 5000 ft
                        cv2.putText(img, f"{y}", 
                            (5, world_to_img_y(y)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                # Draw goal (green circle)
                goal_x = world_to_img_x(self.goal_position[0])
                goal_y = world_to_img_y(self.goal_position[1])
                goal_radius = 15
                
                if 0 <= goal_x < img_size and 0 <= goal_y < img_size:
                    cv2.circle(img, (goal_x, goal_y), goal_radius, (0, 255, 0), -1)
                    cv2.putText(img, "GOAL", (goal_x + 20, goal_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1)
                
                # Draw obstacles (colored circles based on altitude difference)
                for i, obs in enumerate(self.obstacles):
                    obs_x = world_to_img_x(obs["position"][0])
                    obs_y = world_to_img_y(obs["position"][1])
                    
                    # Scale radius to image coordinates
                    obs_radius = int(obs["radius"] / world_width * img_size)
                    
                    # Ensure the radius is visible but not too large
                    obs_radius = max(5, min(obs_radius, 100))
                    
                    if (0 <= obs_x < img_size and 0 <= obs_y < img_size):
                        # Draw with different intensity based on altitude difference
                        alt_diff = abs(obs["position"][2] - self.aircraft["position"][2])
                        
                        if alt_diff < obs["radius"]:  # Dangerous altitude
                            color = (0, 0, 255)  # Red
                            thickness = 2
                        elif alt_diff < obs["radius"] * 2:  # Caution altitude
                            color = (0, 165, 255)  # Orange
                            thickness = 2
                        else:  # Safe altitude
                            color = (180, 180, 180)  # Gray
                            thickness = 1
                            
                        cv2.circle(img, (obs_x, obs_y), obs_radius, color, thickness)
                
                # Draw aircraft (blue triangle oriented by heading)
                ac_x = world_to_img_x(self.aircraft["position"][0])
                ac_y = world_to_img_y(self.aircraft["position"][1])
                heading_rad = np.radians(self.aircraft["heading"])
                
                if 0 <= ac_x < img_size and 0 <= ac_y < img_size:
                    # Triangle points based on heading
                    size = 10
                    pt1_x = ac_x + int(size * 2 * np.sin(heading_rad))
                    pt1_y = ac_y - int(size * 2 * np.cos(heading_rad))
                    pt2_x = ac_x + int(size * np.sin(heading_rad + 2.5))
                    pt2_y = ac_y - int(size * np.cos(heading_rad + 2.5))
                    pt3_x = ac_x + int(size * np.sin(heading_rad - 2.5))
                    pt3_y = ac_y - int(size * np.cos(heading_rad - 2.5))
                    
                    # Create triangle points array
                    pts = np.array([[pt1_x, pt1_y], [pt2_x, pt2_y], [pt3_x, pt3_y]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw filled triangle
                    cv2.fillPoly(img, [pts], (255, 0, 0))
                
                # Add text information
                info_y = 30
                cv2.putText(img, f"Step: {self.current_step}/{self.max_steps}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                pos_str = f"Position: ({self.aircraft['position'][0]:.0f}, {self.aircraft['position'][1]:.0f}, {self.aircraft['position'][2]:.0f})"
                cv2.putText(img, pos_str, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                cv2.putText(img, f"Heading: {self.aircraft['heading']:.1f}°", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                cv2.putText(img, f"Speed: {self.aircraft['speed']:.0f} kts", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                cv2.putText(img, f"VS: {self.aircraft['vertical_speed']:.0f} ft/min", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                goal_dist = np.linalg.norm(self.goal_position - self.aircraft["position"])
                cv2.putText(img, f"Goal dist: {goal_dist:.0f} ft", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                info_y += 25
                
                closest_str = f"Closest approach: {self.closest_to_goal:.0f} ft"
                cv2.putText(img, closest_str, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                # Simple rendering without OpenCV
                # Just showing aircraft and goal as colored squares
                goal_x = world_to_img_x(self.goal_position[0])
                goal_y = world_to_img_y(self.goal_position[1])
                
                if 0 <= goal_x < img_size-10 and 0 <= goal_y < img_size-10:
                    img[goal_y:goal_y+10, goal_x:goal_x+10] = [0, 255, 0]  # Green square for goal
                
                ac_x = world_to_img_x(self.aircraft["position"][0])
                ac_y = world_to_img_y(self.aircraft["position"][1])
                
                if 0 <= ac_x < img_size-10 and 0 <= ac_y < img_size-10:
                    img[ac_y:ac_y+10, ac_x:ac_x+10] = [255, 0, 0]  # Blue square for aircraft
                
                # Draw obstacles as red squares
                for obs in self.obstacles:
                    obs_x = world_to_img_x(obs["position"][0])
                    obs_y = world_to_img_y(obs["position"][1])
                    
                    size = max(5, min(int(obs["radius"] / 200), 20))
                    
                    if (0 <= obs_x < img_size-size and 0 <= obs_y < img_size-size):
                        img[obs_y:obs_y+size, obs_x:obs_x+size] = [0, 0, 255]  # Red square for obstacle
            
            return img
            
        # For 'human' mode, print information to console (only if verbose)
        elif self.render_mode == "human" and self.verbose:
            print(f"Step: {self.current_step}")
            print(f"Position: {self.aircraft['position']}")
            print(f"Heading: {self.aircraft['heading']:.1f}°")
            print(f"Speed: {self.aircraft['speed']:.1f} kts")
            print(f"Vertical Speed: {self.aircraft['vertical_speed']:.1f} ft/min")
            print(f"Distance to goal: {np.linalg.norm(self.goal_position - self.aircraft['position']):.1f} ft")
            print("---")
            return None
        
        return None

    def close(self):
        """Close the environment."""
        pass