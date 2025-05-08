"""
StaticObstacleEnv-v1
====================

A standalone implementation of a 3D air traffic control environment
with static obstacles and vertical speed control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StaticObstacleEnvV1(gym.Env):
    """
    3D Air Traffic Control environment with static obstacles.
    
    This environment simulates an aircraft navigating through 3D airspace
    with static obstacles. The agent must control heading, speed, and vertical speed
    to avoid collisions and reach a goal position.
    
    Action Space:
        3-D continuous vector:
        - [0]: Δ-heading (deg) in range [-45, 45]
        - [1]: Δ-speed (kts) in range [-10, 10]
        - [2]: vertical_speed (ft/min) in range [-2000, 2000]
        
    Observation Space:
        Dictionary with:
        - position: 3D position [x, y, z]
        - heading: Current heading (degrees)
        - speed: Current speed (knots)
        - vertical_speed: Current vertical speed (ft/min)
        - goal_direction: Vector pointing to goal [x, y, z]
        - goal_distance: Distance to goal
        - obstacles: Information about nearby obstacles
        - altitude_diffs: Altitude differences with each obstacle
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, num_obstacles=10, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            num_obstacles: Number of static obstacles
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        print("Initializing standalone StaticObstacleEnv-v1...")
        
        # Environment parameters
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.max_episode_steps = 500
        self.current_step = 0
        self.verbose = True
        
        # Define action space [-45° to 45° heading change, -10 to 10 kts speed change, -2000 to 2000 ft/min VS]
        self.action_space = spaces.Box(
            low=np.array([-45.0, -10.0, -2000.0]),
            high=np.array([45.0, 10.0, 2000.0]),
            dtype=np.float32
        )
        
        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=-10000.0, high=10000.0, shape=(3,), dtype=np.float32),
            "heading": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
            "vertical_speed": spaces.Box(low=-3000.0, high=3000.0, shape=(1,), dtype=np.float32),
            "goal_direction": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "goal_distance": spaces.Box(low=0.0, high=20000.0, shape=(1,), dtype=np.float32),
            "obstacles_info": spaces.Box(low=-10000.0, high=10000.0, shape=(num_obstacles, 4), dtype=np.float32),
            "altitude_diffs": spaces.Box(low=-10000.0, high=10000.0, shape=(num_obstacles,), dtype=np.float32),
        })
        
        # Initialization flag
        self.initialized = False
        
        # Debug info
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
        print("Resetting environment...")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode step counter
        self.current_step = 0
        
        # Initialize aircraft state
        self.aircraft = {
            "position": np.array([0.0, 0.0, 10000.0]),  # Initial position (x, y, altitude in ft)
            "heading": np.array([0.0]),                 # Initial heading (degrees)
            "speed": np.array([250.0]),                 # Initial speed (knots)
            "vertical_speed": np.array([0.0]),          # Initial vertical speed (ft/min)
        }
        
        # Set goal position
        self.goal_position = np.array([8000.0, 8000.0, 10000.0])  # Goal at (x=8000, y=8000, altitude=10000)
        
        # Generate obstacles
        self.obstacles = []
        for i in range(self.num_obstacles):
            # Random position, ensuring obstacles are between aircraft and goal
            x = np.random.uniform(1000.0, 7000.0)
            y = np.random.uniform(1000.0, 7000.0)
            z = np.random.uniform(8000.0, 12000.0)  # Altitude between 8000-12000 ft
            radius = np.random.uniform(300.0, 500.0)  # Collision radius
            
            self.obstacles.append({
                "position": np.array([x, y, z]),
                "radius": radius
            })
        
        # Flag as initialized
        self.initialized = True
        
        # Generate initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        print("Environment reset complete.")
        
        return observation, info

    def step(self, action):
        """
        Apply action and advance the simulation.
        
        Args:
            action: [Δ-heading (deg), Δ-speed (kts), vertical_speed (ft/min)]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure environment is initialized
        if not self.initialized:
            print("WARNING: Environment not initialized. Calling reset()...")
            self.reset()
        
        # Increment step counter
        self.current_step += 1
        
        # Unpack action
        delta_heading = float(action[0])
        delta_speed = float(action[1])
        target_vs = float(action[2])
        
        # Update aircraft state
        # 1. Update heading
        new_heading = float(self.aircraft["heading"]) + delta_heading
        # Keep heading in [0, 360] range
        new_heading = new_heading % 360.0
        self.aircraft["heading"] = np.array([new_heading])
        
        # 2. Update speed
        new_speed = float(self.aircraft["speed"]) + delta_speed
        # Keep speed in reasonable range [150, 350] knots
        new_speed = np.clip(new_speed, 150.0, 350.0)
        self.aircraft["speed"] = np.array([new_speed])
        
        # 3. Set vertical speed
        self.aircraft["vertical_speed"] = np.array([target_vs])
        
        # 4. Update position based on heading, speed and vertical speed
        # Convert heading to radians for movement calculation
        heading_rad = np.radians(new_heading)
        
        # Convert speed from knots to ft/min (1 knot = 101.27 ft/min)
        speed_ftpm = new_speed * 101.27
        
        # Calculate horizontal movement components
        dx = speed_ftpm * np.sin(heading_rad) / 60.0  # per second
        dy = speed_ftpm * np.cos(heading_rad) / 60.0  # per second
        
        # Calculate vertical movement (ft/min to ft/s)
        dz = target_vs / 60.0
        
        # Update position (assuming 1 second time step)
        self.aircraft["position"][0] += dx
        self.aircraft["position"][1] += dy
        self.aircraft["position"][2] += dz
        
        # Constrain altitude to be > 1000 ft
        self.aircraft["position"][2] = max(self.aircraft["position"][2], 1000.0)
        
        # Check for collision with obstacles
        collision = self._check_collision()
        
        # Check if goal reached
        goal_reached = self._check_goal_reached()
        
        # Check for episode end conditions
        terminated = collision or goal_reached
        truncated = self.current_step >= self.max_episode_steps
        
        # Calculate reward
        reward = self._compute_reward(collision, goal_reached)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add additional info
        info["collision"] = collision
        info["goal_reached"] = goal_reached
        info["step"] = self.current_step
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
            dict: The current observation
        """
        # Calculate vector pointing from aircraft to goal
        goal_vector = self.goal_position - self.aircraft["position"]
        goal_distance = np.linalg.norm(goal_vector)
        
        # Normalize goal direction vector
        if goal_distance > 0:
            goal_direction = goal_vector / goal_distance
        else:
            goal_direction = np.zeros(3)
        
        # Create obstacles information array [x, y, z, radius]
        obstacles_info = np.zeros((self.num_obstacles, 4), dtype=np.float32)
        altitude_diffs = np.zeros(self.num_obstacles, dtype=np.float32)
        
        for i, obs in enumerate(self.obstacles):
            obstacles_info[i, :3] = obs["position"]
            obstacles_info[i, 3] = obs["radius"]
            
            # Calculate altitude difference
            altitude_diffs[i] = self.aircraft["position"][2] - obs["position"][2]
        
        return {
            "position": self.aircraft["position"].astype(np.float32),
            "heading": self.aircraft["heading"].astype(np.float32),
            "speed": self.aircraft["speed"].astype(np.float32),
            "vertical_speed": self.aircraft["vertical_speed"].astype(np.float32),
            "goal_direction": goal_direction.astype(np.float32),
            "goal_distance": np.array([goal_distance], dtype=np.float32),
            "obstacles_info": obstacles_info,
            "altitude_diffs": altitude_diffs
        }

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
            "goal_distance": goal_distance,
            "closest_obstacle_distance": closest_distance,
            "aircraft_position": self.aircraft["position"].copy(),
            "aircraft_heading": float(self.aircraft["heading"][0]),
            "aircraft_speed": float(self.aircraft["speed"][0]),
            "aircraft_vs": float(self.aircraft["vertical_speed"][0])
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

    def _check_goal_reached(self):
        """
        Check if aircraft has reached the goal.
        
        Returns:
            bool: True if goal reached, False otherwise
        """
        # Calculate distance to goal
        distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        
        # Consider goal reached if within 500 ft
        return distance < 500.0

    def _compute_reward(self, collision, goal_reached):
        """
        Compute reward based on current state.
        
        Args:
            collision: Whether aircraft has collided
            goal_reached: Whether goal has been reached
            
        Returns:
            float: Reward value
        """
        # Base reward is negative to encourage efficient paths
        reward = -0.1
        
        # Large penalty for collision
        if collision:
            return -100.0
        
        # Large reward for reaching goal
        if goal_reached:
            return 100.0
        
        # Calculate distance to goal
        goal_distance = np.linalg.norm(self.goal_position - self.aircraft["position"])
        
        # Reward inversely proportional to distance (closer = higher reward)
        distance_reward = 10.0 / (1.0 + 0.001 * goal_distance)
        
        # Calculate reward for maintaining safe distance from obstacles
        safety_reward = 0.0
        for obs in self.obstacles:
            # Calculate 3D distance to obstacle
            distance = np.linalg.norm(self.aircraft["position"] - obs["position"])
            
            # If very close to obstacle but not colliding, add small penalty
            if distance < obs["radius"] * 2.0:
                safety_factor = (distance - obs["radius"]) / obs["radius"]
                safety_reward -= 0.5 * (1.0 - safety_factor)
        
        # Calculate efficiency reward (penalize excessive altitude changes)
        vs_penalty = -0.01 * abs(self.aircraft["vertical_speed"][0]) / 1000.0
        
        # Combine rewards
        reward += distance_reward + safety_reward + vs_penalty
        
        return reward

    def render(self):
        """
        Render the environment.
        
        Returns:
            ndarray or None: If render_mode is 'rgb_array', returns an RGB image,
                             otherwise returns None
        """
        if self.render_mode is None:
            return None
            
        # For 'rgb_array' mode, create a simple 2D visualization
        if self.render_mode == "rgb_array":
            # Create a top-down view image
            img_size = 800
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Scale factor to convert world coordinates to image coordinates
            scale = img_size / 16000.0  # Assuming world is roughly ±8000 units
            center = img_size // 2
            
            # Draw goal (green circle)
            goal_x = int(center + self.goal_position[0] * scale)
            goal_y = int(center - self.goal_position[1] * scale)  # Invert y for image coordinates
            goal_radius = 10
            
            if 0 <= goal_x < img_size and 0 <= goal_y < img_size:
                cv_radius = goal_radius
                cv_color = (0, 255, 0)  # Green
                img = self._draw_circle(img, goal_x, goal_y, cv_radius, cv_color)
            
            # Draw obstacles (red circles)
            for obs in self.obstacles:
                obs_x = int(center + obs["position"][0] * scale)
                obs_y = int(center - obs["position"][1] * scale)
                obs_radius = int(obs["radius"] * scale)
                
                if (0 <= obs_x < img_size and 0 <= obs_y < img_size and 
                        obs_radius > 0 and obs_radius < img_size // 2):
                    # Draw with different intensity based on altitude
                    alt_diff = abs(obs["position"][2] - self.aircraft["position"][2])
                    if alt_diff < 1000:  # If within 1000 ft vertically
                        intensity = int(255 * (1 - alt_diff / 1000))
                        cv_color = (0, 0, min(255, intensity + 100))  # More red = more dangerous
                    else:
                        cv_color = (200, 200, 200)  # Gray if far in altitude
                        
                    img = self._draw_circle(img, obs_x, obs_y, obs_radius, cv_color)
            
            # Draw aircraft (blue triangle)
            ac_x = int(center + self.aircraft["position"][0] * scale)
            ac_y = int(center - self.aircraft["position"][1] * scale)
            heading_rad = np.radians(self.aircraft["heading"][0])
            
            if 0 <= ac_x < img_size and 0 <= ac_y < img_size:
                # Triangle points based on heading
                size = 10
                pt1_x = ac_x + int(size * np.sin(heading_rad))
                pt1_y = ac_y - int(size * np.cos(heading_rad))
                pt2_x = ac_x + int(size * np.sin(heading_rad + 2.5))
                pt2_y = ac_y - int(size * np.cos(heading_rad + 2.5))
                pt3_x = ac_x + int(size * np.sin(heading_rad - 2.5))
                pt3_y = ac_y - int(size * np.cos(heading_rad - 2.5))
                
                # Ensure points are within image bounds
                points = [(max(0, min(img_size-1, x)), max(0, min(img_size-1, y))) 
                          for x, y in [(pt1_x, pt1_y), (pt2_x, pt2_y), (pt3_x, pt3_y)]]
                
                # Draw triangle
                for i in range(3):
                    img = self._draw_line(img, *points[i], *points[(i+1)%3], (255, 0, 0))
            
            # Add text information
            img = self._add_text(img, f"Step: {self.current_step}", 10, 20)
            img = self._add_text(img, f"Position: {self.aircraft['position'][0]:.0f}, {self.aircraft['position'][1]:.0f}, {self.aircraft['position'][2]:.0f}", 10, 40)
            img = self._add_text(img, f"Heading: {self.aircraft['heading'][0]:.1f}°", 10, 60)
            img = self._add_text(img, f"Speed: {self.aircraft['speed'][0]:.0f} kts", 10, 80)
            img = self._add_text(img, f"VS: {self.aircraft['vertical_speed'][0]:.0f} ft/min", 10, 100)
            
            return img
            
        # For 'human' mode, print information to console
        elif self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Position: {self.aircraft['position']}")
            print(f"Heading: {self.aircraft['heading'][0]:.1f}°")
            print(f"Speed: {self.aircraft['speed'][0]:.1f} kts")
            print(f"Vertical Speed: {self.aircraft['vertical_speed'][0]:.1f} ft/min")
            print(f"Distance to goal: {np.linalg.norm(self.goal_position - self.aircraft['position']):.1f} ft")
            print("---")
            return None

    def _draw_circle(self, img, x, y, radius, color):
        """Helper method to draw a circle on the image."""
        # Simple implementation without external libraries
        height, width = img.shape[:2]
        for i in range(max(0, x - radius), min(width, x + radius + 1)):
            for j in range(max(0, y - radius), min(height, y + radius + 1)):
                if (i - x)**2 + (j - y)**2 <= radius**2:
                    img[j, i] = color
        return img

    def _draw_line(self, img, x1, y1, x2, y2, color):
        """Helper method to draw a line on the image."""
        # Simple implementation without external libraries
        height, width = img.shape[:2]
        steep = abs(y2 - y1) > abs(x2 - x1)
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        dx = x2 - x1
        dy = abs(y2 - y1)
        error = dx // 2
        y = y1
        y_step = 1 if y1 < y2 else -1
        for x in range(x1, x2 + 1):
            coord = (y, x) if steep else (x, y)
            if 0 <= coord[0] < width and 0 <= coord[1] < height:
                img[coord[1], coord[0]] = color
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        return img

    def _add_text(self, img, text, x, y, color=(0, 0, 0)):
        """Helper method to add text to the image."""
        # Simple implementation - just return the image unchanged
        # In a real implementation, you would use a library like cv2 or PIL
        return img

    def close(self):
        """Close the environment."""
        pass