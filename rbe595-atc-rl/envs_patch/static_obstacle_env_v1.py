from typing import Tuple
import numpy as np
import gymnasium as gym
from bluesky_gym.envs.static_obstacle_env import StaticObstacleEnv
from gymnasium.spaces import Box
from gymnasium import spaces
import math

VS_MIN_FPM, VS_MAX_FPM = -2000.0, 2000.0  # command range ±2,000 ft/min

class StaticObstacleEnvV1(StaticObstacleEnv):
    """
    Adds vertical-speed control and altitude-delta observation.
    Action:  [delta_heading (deg), delta_speed (kts), vertical_speed (ft/min)]
    Obs:     original_obs + [own_alt_ft - obs_alt_ft] for every obstacle
    """
    metadata = StaticObstacleEnv.metadata | {"render_fps": 15}

    def __init__(self):
        super(StaticObstacleEnvV1, self).__init__()

        # Define action space: [heading (-1 to 1 = -45 to 45 deg), speed (0 to 1 = 0 to max_speed), vertical speed (-10 to 10 = scaled ft/min)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -10.0]), 
            high=np.array([1.0, 1.0, 10.0]), 
            dtype=np.float32
        )

        # Define observation space with proper dimensionality
        self.observation_space = spaces.Dict({
            "altitude_diff": spaces.Box(low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32),
            "relative_position": spaces.Box(low=-1000.0, high=1000.0, shape=(2,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "vertical_speed": spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
        })

        # Initialize state
        self.state = {
            "position": np.array([0.0, 0.0]),  # x, y position
            "altitude": np.array([500.0]),     # altitude in feet
            "heading": np.array([0.0]),        # heading in degrees
            "speed": np.array([0.5]),          # normalized speed
            "vertical_speed": np.array([0.0]), # normalized vertical speed
        }
        
        # Environment parameters
        self.max_speed = 250.0  # knots
        self.max_vs = 2000.0    # ft/min
        self.dt = 1.0           # time step in seconds
        
        # Generate obstacles (x, y, altitude)
        self.num_obstacles = 10
        self.obstacles = self.generate_obstacles()
        
        # Goal position
        self.goal_position = np.array([800.0, 800.0])
        self.goal_radius = 50.0
        
        # Episode limits
        self.max_steps = 500
        self.current_step = 0

    def generate_obstacles(self):
        # Generate random obstacles across a 3D space
        # Format: [x, y, altitude]
        obstacles = []
        for _ in range(self.num_obstacles):
            x = np.random.uniform(-500.0, 1000.0)
            y = np.random.uniform(-500.0, 1000.0)
            alt = np.random.uniform(200.0, 800.0)
            obstacles.append(np.array([x, y, alt]))
        return obstacles

    def _get_observation(self):
        # Calculate obstacle altitude differences
        altitude_diffs = []
        for obs in self.obstacles:
            altitude_diffs.append(float(self.state["altitude"] - obs[2]))
        
        # Find closest obstacle for relative position
        closest_idx = self._find_closest_obstacle()
        if closest_idx is not None:
            rel_pos = self.obstacles[closest_idx][:2] - self.state["position"]
        else:
            rel_pos = np.array([1000.0, 1000.0])  # No obstacles nearby
            
        # Create the observation dictionary
        obs = {
            "altitude_diff": np.array([altitude_diffs[0] if altitude_diffs else 0.0], dtype=np.float32),
            "relative_position": np.array(rel_pos, dtype=np.float32),
            "speed": self.state["speed"],
            "vertical_speed": self.state["vertical_speed"],
        }
        return obs

    def _find_closest_obstacle(self):
        if not self.obstacles:
            return None
            
        distances = []
        for i, obs in enumerate(self.obstacles):
            dx = obs[0] - self.state["position"][0]
            dy = obs[1] - self.state["position"][1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances.append((i, dist))
            
        distances.sort(key=lambda x: x[1])
        return distances[0][0]

    def reset(self, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed
            
        # Reset the environment state
        self.state = {
            "position": np.array([0.0, 0.0]),
            "altitude": np.array([500.0]),
            "heading": np.array([0.0]),
            "speed": np.array([0.5]),
            "vertical_speed": np.array([0.0]),
        }
        
        # Reset step counter
        self.current_step = 0
        
        # Generate new obstacles
        self.obstacles = self.generate_obstacles()
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Increment step counter
        self.current_step += 1
        
        # Extract action components
        heading_change, speed_norm, vs_norm = action
        
        # Update heading (convert normalized value to degrees)
        heading_delta = heading_change * 45.0  # -1 to 1 maps to -45 to +45 degrees
        self.state["heading"] = np.array([(self.state["heading"][0] + heading_delta) % 360.0])
        
        # Update speed (normalized between 0 and 1)
        self.state["speed"] = np.array([np.clip(speed_norm, 0.0, 1.0)])
        
        # Update vertical speed (normalized between -10 and 10)
        vs_scaled = vs_norm * (self.max_vs / 10.0)  # Convert to actual ft/min
        self.state["vertical_speed"] = np.array([vs_norm])
        
        # Calculate actual speed in knots
        speed_kts = float(self.state["speed"] * self.max_speed)
        
        # Convert heading to radians for position update
        heading_rad = np.radians(float(self.state["heading"]))
        
        # Update position based on heading and speed
        dx = speed_kts * np.sin(heading_rad) * self.dt / 60.0  # Convert to distance per time step
        dy = speed_kts * np.cos(heading_rad) * self.dt / 60.0
        self.state["position"][0] += dx
        self.state["position"][1] += dy
        
        # Update altitude based on vertical speed
        altitude_change = vs_scaled * self.dt / 60.0  # ft/min to ft per time step
        self.state["altitude"] = np.array([self.state["altitude"][0] + altitude_change])
        
        # Check for collisions or goal reached
        collision = self._check_collision()
        goal_reached = self._check_goal()
        timeout = self.current_step >= self.max_steps
        
        # Determine if episode is done
        terminated = collision or goal_reached
        truncated = timeout
        
        # Calculate reward
        reward = self._compute_reward(collision, goal_reached)
        
        # Get updated observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "position": self.state["position"],
            "altitude": float(self.state["altitude"][0]),
            "step": self.current_step
        }
        
        return observation, reward, terminated, truncated, info

    def _check_collision(self):
        # Check for collisions with obstacles
        for obs in self.obstacles:
            # Calculate 3D distance
            dx = obs[0] - self.state["position"][0]
            dy = obs[1] - self.state["position"][1]
            dz = obs[2] - float(self.state["altitude"])
            
            # Check for horizontal proximity
            horizontal_dist = np.sqrt(dx*dx + dy*dy)
            if horizontal_dist < 50.0:  # 50 unit safety radius
                # Check for vertical proximity
                if abs(dz) < 100.0:  # 100 ft vertical separation
                    return True
        return False

    def _check_goal(self):
        # Check if aircraft has reached the goal
        dx = self.goal_position[0] - self.state["position"][0]
        dy = self.goal_position[1] - self.state["position"][1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance < self.goal_radius

    def _compute_reward(self, collision, goal_reached):
        if collision:
            return -100.0  # Large penalty for collision
            
        if goal_reached:
            return 100.0   # Large reward for reaching goal
            
        # Calculate distance to goal for reward shaping
        dx = self.goal_position[0] - self.state["position"][0]
        dy = self.goal_position[1] - self.state["position"][1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Reward is inversely proportional to distance
        distance_reward = -0.1 * distance / 1000.0
        
        # Small penalty for each step to encourage efficiency
        step_penalty = -0.1
        
        return distance_reward + step_penalty

    def render(self):
        # Simple text-based rendering (optional)
        print(f"Position: {self.state['position']}, "
              f"Altitude: {float(self.state['altitude'][0]):.1f}, "
              f"Heading: {float(self.state['heading'][0]):.1f}°, "
              f"Speed: {float(self.state['speed'][0]):.2f}, "
              f"VS: {float(self.state['vertical_speed'][0]):.2f}")
              
        # If you want to return an image for video recording:
        # This is a very simple visualization - you can enhance it
        img_size = 500
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Scale coordinates to image size
        scale = 0.25
        offset = img_size // 2
        
        # Draw obstacles (red circles)
        for obs in self.obstacles:
            x = int(obs[0] * scale + offset)
            y = int(obs[1] * scale + offset)
            if 0 <= x < img_size and 0 <= y < img_size:
                # Draw a red circle
                cv_radius = 5
                x1, y1 = max(0, x - cv_radius), max(0, y - cv_radius)
                x2, y2 = min(img_size-1, x + cv_radius), min(img_size-1, y + cv_radius)
                img[y1:y2, x1:x2] = [0, 0, 255]  # BGR: Red
        
        # Draw agent (green circle)
        agent_x = int(self.state["position"][0] * scale + offset)
        agent_y = int(self.state["position"][1] * scale + offset)
        if 0 <= agent_x < img_size and 0 <= agent_y < img_size:
            cv_radius = 7
            x1, y1 = max(0, agent_x - cv_radius), max(0, agent_y - cv_radius)
            x2, y2 = min(img_size-1, agent_x + cv_radius), min(img_size-1, agent_y + cv_radius)
            img[y1:y2, x1:x2] = [0, 255, 0]  # BGR: Green
            
        # Draw goal (blue circle)
        goal_x = int(self.goal_position[0] * scale + offset)
        goal_y = int(self.goal_position[1] * scale + offset)
        if 0 <= goal_x < img_size and 0 <= goal_y < img_size:
            cv_radius = 10
            x1, y1 = max(0, goal_x - cv_radius), max(0, goal_y - cv_radius)
            x2, y2 = min(img_size-1, goal_x + cv_radius), min(img_size-1, goal_y + cv_radius)
            img[y1:y2, x1:x2] = [255, 0, 0]  # BGR: Blue
        
        return img