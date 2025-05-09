"""
SectorCREnv-v1
==============

A standalone implementation of a 3D air traffic control environment
for sector conflict resolution with vertical speed control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SectorCREnvV1(gym.Env):
    """
    3D Air Traffic Control environment for sector conflict resolution.
    
    This environment simulates an aircraft navigating through 3D airspace
    with multiple intruder aircraft. The agent must control heading, speed,
    and vertical speed to avoid conflicts while maintaining an efficient path.
    
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
        - intruders: Information about intruder aircraft
        - altitude_diffs: Altitude differences with each intruder
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, num_intruders=3, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            num_intruders: Number of intruder aircraft
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        print("Initializing standalone SectorCREnv-v1...")
        
        # Environment parameters
        self.num_intruders = num_intruders
        self.render_mode = render_mode
        self.max_episode_steps = 500
        self.current_step = 0
        self.verbose = True
        self.agent_callsign = "OWN"  # Own aircraft callsign
        
        # Define separation minima
        self.horizontal_separation_min = 3.0  # 3 nautical miles
        self.vertical_separation_min = 1000.0  # 1000 feet
        
        # Define sector boundaries
        self.sector_dims = {
            "x_min": -50.0, "x_max": 50.0,  # nautical miles
            "y_min": -50.0, "y_max": 50.0,  # nautical miles
            "z_min": 10000.0, "z_max": 40000.0  # feet
        }
        
        # Define action space [-45° to 45° heading change, -10 to 10 kts speed change, -2000 to 2000 ft/min VS]
        self.action_space = spaces.Box(
            low=np.array([-45.0, -10.0, -2000.0]),
            high=np.array([45.0, 10.0, 2000.0]),
            dtype=np.float32
        )
        
        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([self.sector_dims["x_min"], self.sector_dims["y_min"], self.sector_dims["z_min"]]),
                high=np.array([self.sector_dims["x_max"], self.sector_dims["y_max"], self.sector_dims["z_max"]]),
                dtype=np.float32
            ),
            "heading": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=100.0, high=500.0, shape=(1,), dtype=np.float32),
            "vertical_speed": spaces.Box(low=-3000.0, high=3000.0, shape=(1,), dtype=np.float32),
            "goal_direction": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "goal_distance": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "intruders_info": spaces.Box(
                low=np.tile(
                    np.array([
                        self.sector_dims["x_min"], self.sector_dims["y_min"], self.sector_dims["z_min"],
                        -500.0, -500.0, -500.0, 0.0, 0.0, -3000.0
                    ]),
                    (num_intruders, 1)
                ).reshape(num_intruders, 9),
                high=np.tile(
                    np.array([
                        self.sector_dims["x_max"], self.sector_dims["y_max"], self.sector_dims["z_max"],
                        500.0, 500.0, 500.0, 360.0, 500.0, 3000.0
                    ]),
                    (num_intruders, 1)
                ).reshape(num_intruders, 9),
                dtype=np.float32
            ),
            "altitude_diffs": spaces.Box(low=-30000.0, high=30000.0, shape=(num_intruders,), dtype=np.float32),
        })
        
        # Initialize prev_goal_distance for reward calculation
        self.prev_goal_distance = None
        
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
        
        # Initialize own aircraft state
        self.own_aircraft = {
            "position": np.array([0.0, 0.0, 20000.0]),  # Initial position (x, y in NM, altitude in ft)
            "heading": np.array([45.0]),                 # Initial heading (degrees)
            "speed": np.array([250.0]),                 # Initial speed (knots)
            "vertical_speed": np.array([0.0]),          # Initial vertical speed (ft/min)
            "callsign": self.agent_callsign
        }
        
        # Set goal position (exit point from sector)
        self.goal_position = np.array([40.0, 40.0, 20000.0])  # Goal position
        
        # Reset previous goal distance for reward calculation
        self.prev_goal_distance = np.linalg.norm(self.goal_position - self.own_aircraft["position"])
        
        # Generate intruder aircraft
        self.intruders = []
        for i in range(self.num_intruders):
            # Random position within sector
            x = np.random.uniform(self.sector_dims["x_min"] * 0.8, self.sector_dims["x_max"] * 0.8)
            y = np.random.uniform(self.sector_dims["y_min"] * 0.8, self.sector_dims["y_max"] * 0.8)
            z = np.random.uniform(15000.0, 35000.0)  # Altitude
            
            # Random heading, speed, and vertical speed
            heading = np.random.uniform(0.0, 360.0)
            speed = np.random.uniform(200.0, 300.0)  # knots
            vs = np.random.choice([-500.0, 0.0, 500.0])  # ft/min
            
            # Create intruder
            intruder = {
                "position": np.array([x, y, z]),
                "heading": heading,
                "speed": speed,
                "vertical_speed": vs,
                "callsign": f"INTR{i+1}"
            }
            
            self.intruders.append(intruder)
        
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
        
        # Update own aircraft state
        # 1. Update heading
        new_heading = float(self.own_aircraft["heading"]) + delta_heading
        # Keep heading in [0, 360] range
        new_heading = new_heading % 360.0
        self.own_aircraft["heading"] = np.array([new_heading])
        
        # 2. Update speed
        new_speed = float(self.own_aircraft["speed"]) + delta_speed
        # Keep speed in reasonable range [150, 350] knots
        new_speed = np.clip(new_speed, 150.0, 350.0)
        self.own_aircraft["speed"] = np.array([new_speed])
        
        # 3. Set vertical speed
        self.own_aircraft["vertical_speed"] = np.array([target_vs])
        
        # 4. Update position based on heading, speed and vertical speed
        self._update_aircraft_position(self.own_aircraft)
        
        # Update intruder aircraft positions
        for intruder in self.intruders:
            self._update_aircraft_position(intruder)
        
        # Check for conflicts with intruders
        conflict = self._check_conflict()
        
        # Check if goal reached
        goal_reached = self._check_goal_reached()
        
        # Check if aircraft is outside sector boundaries
        out_of_bounds = self._check_out_of_bounds(self.own_aircraft)
        
        # Check for episode end conditions
        terminated = conflict or goal_reached or out_of_bounds
        truncated = self.current_step >= self.max_episode_steps
        
        # Calculate reward
        reward = self._compute_reward(conflict, goal_reached, out_of_bounds)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add additional info
        info["conflict"] = conflict
        info["goal_reached"] = goal_reached
        info["out_of_bounds"] = out_of_bounds
        info["step"] = self.current_step
        
        return observation, reward, terminated, truncated, info

    def _update_aircraft_position(self, aircraft):
        """
        Update aircraft position based on heading, speed and vertical speed.
        
        Args:
            aircraft: Aircraft data dictionary
        """
        # Convert heading to radians for movement calculation
        heading_rad = np.radians(aircraft["heading"] if isinstance(aircraft["heading"], float) else aircraft["heading"][0])
        
        # Get speed in knots
        speed = aircraft["speed"] if isinstance(aircraft["speed"], float) else aircraft["speed"][0]
        
        # Get vertical speed in ft/min
        vs = aircraft["vertical_speed"] if isinstance(aircraft["vertical_speed"], float) else aircraft["vertical_speed"][0]
        
        # Calculate horizontal movement (assuming 1 minute time step, converting to NM)
        # 1 knot = 1 NM per hour, so for 1 minute we divide by 60
        dx = speed * np.sin(heading_rad) / 60.0
        dy = speed * np.cos(heading_rad) / 60.0
        
        # Calculate vertical movement (ft/min for 1 minute = ft)
        dz = vs
        
        # Update position
        aircraft["position"][0] += dx
        aircraft["position"][1] += dy
        aircraft["position"][2] += dz
        
        # Constrain altitude to be within sector bounds
        aircraft["position"][2] = np.clip(
            aircraft["position"][2],
            self.sector_dims["z_min"],
            self.sector_dims["z_max"]
        )

    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
            dict: The current observation
        """
        # Calculate vector pointing from aircraft to goal
        goal_vector = self.goal_position - self.own_aircraft["position"]
        goal_distance = np.linalg.norm(goal_vector)
        
        # Normalize goal direction vector
        if goal_distance > 0:
            goal_direction = goal_vector / goal_distance
        else:
            goal_direction = np.zeros(3)
        
        # Create intruders information array
        intruders_info = np.zeros((self.num_intruders, 9), dtype=np.float32)
        altitude_diffs = np.zeros(self.num_intruders, dtype=np.float32)
        
        for i, intruder in enumerate(self.intruders):
            # Absolute position
            intruders_info[i, 0:3] = intruder["position"]
            
            # Relative position to own aircraft
            rel_position = intruder["position"] - self.own_aircraft["position"]
            intruders_info[i, 3:6] = rel_position
            
            # Heading, speed, vertical speed
            intruders_info[i, 6] = intruder["heading"] if isinstance(intruder["heading"], float) else intruder["heading"][0]
            intruders_info[i, 7] = intruder["speed"] if isinstance(intruder["speed"], float) else intruder["speed"][0]
            intruders_info[i, 8] = intruder["vertical_speed"] if isinstance(intruder["vertical_speed"], float) else intruder["vertical_speed"][0]
            
            # Calculate altitude difference
            altitude_diffs[i] = self.own_aircraft["position"][2] - intruder["position"][2]
        
        return {
            "position": self.own_aircraft["position"].astype(np.float32),
            "heading": self.own_aircraft["heading"].astype(np.float32),
            "speed": self.own_aircraft["speed"].astype(np.float32),
            "vertical_speed": self.own_aircraft["vertical_speed"].astype(np.float32),
            "goal_direction": goal_direction.astype(np.float32),
            "goal_distance": np.array([goal_distance], dtype=np.float32),
            "intruders_info": intruders_info,
            "altitude_diffs": altitude_diffs
        }

    def _get_info(self):
        """
        Get additional information about the environment state.
        
        Returns:
            dict: Additional information
        """
        # Calculate distance to goal
        goal_distance = np.linalg.norm(self.goal_position - self.own_aircraft["position"])
        
        # Check for conflicts and calculate closest intruder distance
        min_horizontal_separation = float('inf')
        min_vertical_separation = float('inf')
        
        for intruder in self.intruders:
            # Calculate horizontal distance (in NM)
            dx = self.own_aircraft["position"][0] - intruder["position"][0]
            dy = self.own_aircraft["position"][1] - intruder["position"][1]
            horizontal_distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate vertical distance (in feet)
            vertical_distance = abs(self.own_aircraft["position"][2] - intruder["position"][2])
            
            # Update minimum separations
            min_horizontal_separation = min(min_horizontal_separation, horizontal_distance)
            min_vertical_separation = min(min_vertical_separation, vertical_distance)
        
        return {
            "goal_distance": goal_distance,
            "min_horizontal_separation": min_horizontal_separation,
            "min_vertical_separation": min_vertical_separation,
            "aircraft_position": self.own_aircraft["position"].copy(),
            "aircraft_heading": float(self.own_aircraft["heading"][0]),
            "aircraft_speed": float(self.own_aircraft["speed"][0]),
            "aircraft_vs": float(self.own_aircraft["vertical_speed"][0])
        }

    def _check_conflict(self):
        """
        Check if aircraft has a conflict with any intruder.
        
        A conflict occurs when horizontal separation is less than minimum
        AND vertical separation is less than minimum.
        
        Returns:
            bool: True if conflict exists, False otherwise
        """
        for intruder in self.intruders:
            # Calculate horizontal distance (in NM)
            dx = self.own_aircraft["position"][0] - intruder["position"][0]
            dy = self.own_aircraft["position"][1] - intruder["position"][1]
            horizontal_distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate vertical distance (in feet)
            vertical_distance = abs(self.own_aircraft["position"][2] - intruder["position"][2])
            
            # Check for conflict
            if (horizontal_distance < self.horizontal_separation_min and 
                vertical_distance < self.vertical_separation_min):
                return True
                
        return False

    def _check_goal_reached(self):
        """
        Check if aircraft has reached the goal.
        
        Returns:
            bool: True if goal reached, False otherwise
        """
        # Calculate distance to goal
        distance = np.linalg.norm(self.goal_position - self.own_aircraft["position"])
        
        # Consider goal reached if within 3 NM
        return distance < 3.0

    def _check_out_of_bounds(self, aircraft):
        """
        Check if aircraft is outside sector boundaries.
        
        Args:
            aircraft: Aircraft data dictionary
            
        Returns:
            bool: True if out of bounds, False otherwise
        """
        pos = aircraft["position"]
        
        return (pos[0] < self.sector_dims["x_min"] or pos[0] > self.sector_dims["x_max"] or
                pos[1] < self.sector_dims["y_min"] or pos[1] > self.sector_dims["y_max"] or
                pos[2] < self.sector_dims["z_min"] or pos[2] > self.sector_dims["z_max"])

    def _compute_reward(self, conflict, goal_reached, out_of_bounds):
        """
        Compute reward based on current state with improved goal-directed incentives.
        """
        # Base reward is slightly negative to encourage efficient paths
        reward = -0.1
        
        # Terminal rewards/penalties
        if conflict:
            return -100.0  # Keep the same penalty for conflict
        
        if out_of_bounds:
            return -50.0  # Keep the same penalty for out of bounds
        
        if goal_reached:
            return 100.0  # Keep the same reward for reaching goal
        
        # Calculate distance to goal
        goal_distance = np.linalg.norm(self.goal_position - self.own_aircraft["position"])
        
        # Store previous step's distance to calculate progress
        if self.prev_goal_distance is None:
            self.prev_goal_distance = goal_distance
        
        # Calculate progress toward goal (positive = getting closer)
        progress = self.prev_goal_distance - goal_distance
        self.prev_goal_distance = goal_distance
        
        # Higher reward for making progress toward goal
        progress_reward = progress * 15.0  # Strongly incentivize moving toward goal
        
        # Reward inversely proportional to distance (closer = higher reward)
        # Make this reward stronger to create a stronger pull toward the goal
        distance_reward = 10.0 / (1.0 + 0.1 * goal_distance)  # Increased from 5.0
        
        # Add directional reward - reward when heading toward goal
        heading_rad = np.radians(float(self.own_aircraft["heading"][0]))
        goal_vec = self.goal_position - self.own_aircraft["position"]
        goal_angle = np.arctan2(goal_vec[0], goal_vec[1])
        heading_diff = abs(heading_rad - goal_angle) % (2 * np.pi)
        if heading_diff > np.pi:
            heading_diff = 2 * np.pi - heading_diff
        
        # Reward for aligning heading with goal direction
        # This will encourage the agent to point toward the goal
        direction_reward = 3.0 * (1.0 - heading_diff / np.pi)
        
        # Reward for maintaining separation from intruders
        separation_reward = 0.0
        for intruder in self.intruders:
            # Calculate horizontal distance (in NM)
            dx = self.own_aircraft["position"][0] - intruder["position"][0]
            dy = self.own_aircraft["position"][1] - intruder["position"][1]
            horizontal_distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate vertical distance (in feet)
            vertical_distance = abs(self.own_aircraft["position"][2] - intruder["position"][2])
            
            # If getting close to minimum separation, add penalty
            h_margin = horizontal_distance - self.horizontal_separation_min
            v_margin = vertical_distance - self.vertical_separation_min
            
            if h_margin < 3.0 and v_margin < 1000.0:
                # Calculate safety margin factor (0 = at minimum separation, 1 = far away)
                h_factor = min(1.0, h_margin / 3.0)
                v_factor = min(1.0, v_margin / 1000.0)
                
                # Lower reward for closer approaches
                separation_reward -= (1.0 - h_factor) * (1.0 - v_factor) * 2.0
        
        # Calculate efficiency reward (penalize excessive heading changes and altitude changes)
        efficiency_reward = 0.0
        
        # Penalize excessive vertical speed
        vs_magnitude = abs(float(self.own_aircraft["vertical_speed"][0]))
        if vs_magnitude > 500.0:
            efficiency_reward -= (vs_magnitude - 500.0) / 1500.0
        
        # Combine rewards - note the increased weights for goal-directed rewards
        reward += direction_reward + progress_reward + distance_reward + separation_reward + efficiency_reward
        
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
            scale = img_size / (self.sector_dims["x_max"] - self.sector_dims["x_min"])
            offset_x = -self.sector_dims["x_min"] * scale
            offset_y = -self.sector_dims["y_min"] * scale
            
            # Draw sector boundaries
            # ... (implementation would draw lines for sector boundaries)
            
            # Draw goal (green circle)
            goal_x = int(self.goal_position[0] * scale + offset_x)
            goal_y = int((self.sector_dims["y_max"] - self.goal_position[1]) * scale)  # Invert y for image coordinates
            goal_radius = 10
            
            if 0 <= goal_x < img_size and 0 <= goal_y < img_size:
                cv_radius = goal_radius
                cv_color = (0, 255, 0)  # Green
                img = self._draw_circle(img, goal_x, goal_y, cv_radius, cv_color)
            
            # Draw intruders (red circles with shade based on vertical separation)
            for intruder in self.intruders:
                intr_x = int(intruder["position"][0] * scale + offset_x)
                intr_y = int((self.sector_dims["y_max"] - intruder["position"][1]) * scale)
                
                if 0 <= intr_x < img_size and 0 <= intr_y < img_size:
                    # Draw with different intensity based on altitude difference
                    alt_diff = abs(intruder["position"][2] - self.own_aircraft["position"][2])
                    if alt_diff < self.vertical_separation_min:
                        intensity = int(255 * (1 - alt_diff / self.vertical_separation_min))
                        cv_color = (0, 0, min(255, intensity + 100))  # More red = more dangerous
                    else:
                        cv_color = (200, 200, 200)  # Gray if far in altitude
                        
                    img = self._draw_circle(img, intr_x, intr_y, 8, cv_color)
            
            # Draw own aircraft (blue triangle)
            ac_x = int(self.own_aircraft["position"][0] * scale + offset_x)
            ac_y = int((self.sector_dims["y_max"] - self.own_aircraft["position"][1]) * scale)
            heading_rad = np.radians(self.own_aircraft["heading"][0])
            
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
            img = self._add_text(img, f"Position: {self.own_aircraft['position'][0]:.1f}, {self.own_aircraft['position'][1]:.1f}, {self.own_aircraft['position'][2]:.0f}", 10, 40)
            img = self._add_text(img, f"Heading: {self.own_aircraft['heading'][0]:.1f}°", 10, 60)
            img = self._add_text(img, f"Speed: {self.own_aircraft['speed'][0]:.0f} kts", 10, 80)
            img = self._add_text(img, f"VS: {self.own_aircraft['vertical_speed'][0]:.0f} ft/min", 10, 100)
            
            return img
            
        # For 'human' mode, print information to console
        elif self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Position: {self.own_aircraft['position']}")
            print(f"Heading: {self.own_aircraft['heading'][0]:.1f}°")
            print(f"Speed: {self.own_aircraft['speed'][0]:.1f} kts")
            print(f"Vertical Speed: {self.own_aircraft['vertical_speed'][0]:.1f} ft/min")
            print(f"Distance to goal: {np.linalg.norm(self.goal_position - self.own_aircraft['position']):.1f}")
            
            # Print intruder information
            print("\nIntruders:")
            for i, intruder in enumerate(self.intruders):
                dx = self.own_aircraft["position"][0] - intruder["position"][0]
                dy = self.own_aircraft["position"][1] - intruder["position"][1]
                dz = self.own_aircraft["position"][2] - intruder["position"][2]
                h_dist = np.sqrt(dx**2 + dy**2)
                print(f"  #{i+1}: H-dist: {h_dist:.1f} NM, V-dist: {abs(dz):.0f} ft")
            
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