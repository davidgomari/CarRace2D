# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pygame # For rendering and human input
import math

from track import OvalTrack
from car import Car
from utils import check_collision
from agents.human_agent import HumanAgent # Import agent types
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from agents.mpc_agent import MPCAgent
from ui import UI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RacingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config, mode='multi'):
        super().__init__()
        self.mode = mode
        self._load_config(config)
        self._setup_track()
        self._setup_spaces() # Call before _setup_agents_and_cars if they need spaces
        
        # Pre-calculate centerline points for projection if track is OvalTrack
        self._discretized_centerline_cache = []
        self._centerline_projection_segments = 200 # Number of points to discretize centerline for projection
        if isinstance(self.track, OvalTrack):
            total_s = self.track.centerline_length
            for i in range(self._centerline_projection_segments):
                s = (i / self._centerline_projection_segments) * total_s
                x, y, theta = self._get_oval_centerline_point(s)
                self._discretized_centerline_cache.append({'s': s, 'x': x, 'y': y, 'theta': theta})

        self._setup_agents_and_cars()
        self.quit_requested = False # Flag to signal user quit
        
        # Initialize UI if needed
        self.ui = None
        if self.render_mode in ['human', 'rgb_array']:
            self.ui = UI(config, self.track)

    def _load_config(self, config):
        """Load and store configuration from YAML file."""
        self.config = config
        sim_cfg = self.config['simulation']
        self.dt = sim_cfg['dt']
        self.max_steps = sim_cfg['max_steps']
        self.render_mode = sim_cfg['render_mode']
        self.current_step = 0
        self.num_agents = sim_cfg['num_agents'] # Will be adjusted in _setup_agents_and_cars for single mode

        # Load observation components config
        env_cfg = self.config.get('environment', {})
        self.observation_components = env_cfg.get('observation_components', ['x', 'y', 'v', 'theta'])
        print(f"RL Agent observation components: {self.observation_components}")

        # LiDAR configuration
        car_cfg = self.config.get('car', {})
        self.lidar_num_beams = car_cfg.get('lidar_num_beams', 32)
        self.lidar_max_range = car_cfg.get('lidar_max_range', 50.0)
        self.lidar_eps = float(car_cfg.get('lidar_eps', 1e-3))
        
        # Default target speed for MPC reference path (as a percentage of max_speed)
        self.mpc_ref_path_target_speed_factor = self.config.get('environment', {}).get('mpc_ref_path_target_speed_factor', 0.75)


    def _setup_track(self):
        """Initialize the racing track."""
        track_cfg = self.config['track']
        if track_cfg['type'] == 'oval':
            self.track = OvalTrack(track_cfg)
        else:
            raise ValueError(f"Unknown track type: {track_cfg['type']}")
        
        # Add track parameters to car config for lap checking
        self.config['car']['track_half_width'] = self.track.half_width
        self.config['car']['track_radius'] = self.track.radius

    # --- Helper methods for Oval Track Centerline and MPC Reference Path ---
    def _get_oval_centerline_point(self, s_raw):
        """
        Calculates the (x, y, theta) coordinates and tangent angle 
        on the oval track's centerline for a given arc length s_raw.
        Assumes s=0 starts at the beginning of the first straight segment.
        """
        L = self.track.length
        R = self.track.radius
        total_length = self.track.centerline_length
        s = s_raw % total_length # Ensure s is within [0, total_length)

        # Define critical points (arc lengths at segment transitions)
        s_bottom_straight_end = L
        s_right_curve_end = L + np.pi * R
        s_top_straight_end = L + np.pi * R + L
        # s_left_curve_end = total_length

        if 0 <= s < s_bottom_straight_end: # Bottom straight
            prog_in_seg = s
            x = -L/2 + prog_in_seg
            y = -R
            theta = 0.0
        elif s_bottom_straight_end <= s < s_right_curve_end: # Right curve
            prog_in_seg = s - s_bottom_straight_end
            angle_in_curve = prog_in_seg / R 
            # Curve starts at (-pi/2) relative to curve center's +x axis
            current_angle_on_circle = -np.pi/2 + angle_in_curve 
            x = self.track.curve1_center_x + R * np.cos(current_angle_on_circle)
            y = self.track.curve1_center_y + R * np.sin(current_angle_on_circle)
            theta = current_angle_on_circle + np.pi/2
        elif s_right_curve_end <= s < s_top_straight_end: # Top straight
            prog_in_seg = s - s_right_curve_end
            x = L/2 - prog_in_seg
            y = R
            theta = np.pi
        else: # Left curve (s_top_straight_end <= s < total_length)
            prog_in_seg = s - s_top_straight_end
            angle_in_curve = prog_in_seg / R
            # Curve starts at (pi/2) relative to curve center's +x axis
            current_angle_on_circle = np.pi/2 + angle_in_curve
            x = self.track.curve2_center_x + R * np.cos(current_angle_on_circle)
            y = self.track.curve2_center_y + R * np.sin(current_angle_on_circle)
            theta = current_angle_on_circle + np.pi/2

        theta = (theta + np.pi) % (2 * np.pi) - np.pi # Normalize theta to [-pi, pi]
        return x, y, theta

    def _project_to_oval_centerline(self, car_x, car_y):
        """
        Projects the car's (x,y) position to the closest point on the pre-calculated
        discretized centerline.
        Returns: (s_proj, x_proj, y_proj, theta_proj, signed_dist_to_centerline)
        """
        if not self._discretized_centerline_cache:
            # Fallback if cache is not populated (should not happen with OvalTrack)
            # This would be a crude projection, ideally log an error or handle better
            print("Warning: Centerline cache not populated for projection.")
            # A very rough estimate if needed: use track's own dist_to_centerline if it provided more
            # For now, return a default that indicates failure to project properly
            return 0.0, car_x, car_y, 0.0, self.track.get_distance_to_centerline(car_x, car_y)


        min_dist_sq = float('inf')
        closest_point_info = None

        for point_info in self._discretized_centerline_cache:
            dist_sq = (car_x - point_info['x'])**2 + (car_y - point_info['y'])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point_info = point_info
        
        s_proj = closest_point_info['s']
        x_proj = closest_point_info['x']
        y_proj = closest_point_info['y']
        theta_proj = closest_point_info['theta']

        # Calculate signed distance
        # Vector from projected centerline point to car
        vec_proj_to_car_x = car_x - x_proj
        vec_proj_to_car_y = car_y - y_proj
        # Left normal vector of the centerline tangent
        normal_x = -np.sin(theta_proj)
        normal_y = np.cos(theta_proj)
        # Dot product gives signed distance (positive if car is to the left of centerline)
        signed_dist = vec_proj_to_car_x * normal_x + vec_proj_to_car_y * normal_y
        
        return s_proj, x_proj, y_proj, theta_proj, signed_dist

    def get_curvature_at_s(self, s_raw):
        """Approximates centerline curvature at arc length s_raw."""
        ds = 0.5  # Slightly larger ds for stability in theta calculation
        # Ensure s_raw is not at the very start or end if using ds/2 offsets without wrapping s
        total_len = self.track.centerline_length
        s1 = (s_raw - ds / 2 + total_len) % total_len # Handle wrapping
        s2 = (s_raw + ds / 2) % total_len

        _, _, theta1 = self._get_oval_centerline_point(s1)
        _, _, theta2 = self._get_oval_centerline_point(s2)

        dtheta = theta2 - theta1
        # Normalize angle difference to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        curvature = dtheta / ds if abs(ds) > 1e-6 else 0.0
        return curvature

    def _generate_reference_path(self, s_current, current_car_speed, mpc_agent, car_config):
        N = mpc_agent.N
        dt_mpc = mpc_agent.dt
        reference_path = np.zeros((N, 4))
        max_speed_car = car_config.get('max_speed', 20.0)

        # Original target speed factor
        base_target_v_ref = max_speed_car * self.mpc_ref_path_target_speed_factor

        # Max lateral acceleration for reference speed calculation (e.g., 70% of tire friction limit)
        # Use car_config directly as it's passed in
        max_lat_accel_for_ref_speed = car_config.get('coeff_friction', 1.1) * \
                                    car_config.get('gravity', 9.81) * \
                                    0.7  # Safety factor for reference speed

        current_s_on_path = s_current
        for k in range(N):
            # Predict s for the next point on the reference path
            # Using a blend of current speed and a moderate lookahead speed for s progression
            # to prevent excessive lookahead if current_car_speed is very high.
            lookahead_speed_for_s = min(current_car_speed, base_target_v_ref * 1.2) # Cap lookahead advance speed
            lookahead_dist_increment = lookahead_speed_for_s * dt_mpc
            current_s_on_path += lookahead_dist_increment

            x_ref, y_ref, theta_ref = self._get_oval_centerline_point(current_s_on_path)

            curvature_val = abs(self.get_curvature_at_s(current_s_on_path))

            v_ref_curvature_limit = max_speed_car
            if curvature_val > 1e-4: # Avoid division by zero or extreme speeds on straights
                v_ref_curvature_limit = np.sqrt(max_lat_accel_for_ref_speed / curvature_val)

            # Effective reference speed for this point
            final_v_ref = min(base_target_v_ref, v_ref_curvature_limit)
            final_v_ref = max(final_v_ref, 1.0) # Ensure a minimum positive speed

            reference_path[k, :] = [x_ref, y_ref, theta_ref, final_v_ref]

        return reference_path
    # --- End of MPC Helper Methods ---

    def _setup_agents_and_cars(self):
        """Initialize agents and their corresponding cars."""
        self.agents = {}
        self.cars = {}
        agent_colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'gray', 'pink', 'brown', 'lime']

        # Determine the correct num_agents based on mode and config
        if self.mode == 'single':
            self.num_agents = 1
            # Ensure config reflects single agent if mode is single
            if 'agent_0' not in self.config.get('agents', {}):
                 self.config['agents'] = {'agent_0': {'type': 'human', 'start_pos_idx': 0}} # Default fallback
            self.config['agents'] = {'agent_0': self.config['agents']['agent_0']}
        else:
            # Use num_agents from multi-agent config or count defined agents
            self.num_agents = self.config.get('simulation', {}).get('num_agents', len(self.config.get('agents', {})))
            if self.num_agents == 0:
                raise ValueError("Multi-agent mode selected but no agents defined in config or num_agents is 0.")
        
        start_positions = self.track.get_starting_positions(self.num_agents)
        agent_keys = list(self.config.get('agents', {}).keys())[:self.num_agents]
        
        if len(agent_keys) != self.num_agents:
             print(f"Warning: Number of agents in config ({len(agent_keys)}) does not match expected num_agents ({self.num_agents}). Using {self.num_agents} agents.")

        for i in range(self.num_agents):
            # Ensure we use expected agent_ids like 'agent_0', 'agent_1', ... for consistency
            agent_id = f"agent_{i}"
            # If the config has a different key name at this index, map it.
            # This assumes the order in the config file matters for multi-agent.
            if 1 < len(agent_keys):
                 config_agent_key = agent_keys[i]
                 if config_agent_key != agent_id:
                      print(f"Mapping config agent key '{config_agent_key}' to runtime ID '{agent_id}'")
                      self.config['agents'][agent_id] = self.config['agents'].pop(config_agent_key)
                 # Now use agent_id guaranteed
            # else:
            #      # This case should ideally not happen if config validation is good
            #      print(f"Warning: Configuration missing for agent index {i}. Creating default agent.")
            #      self.config['agents'][agent_id] = {'type': 'random', 'start_pos_idx': i} # Default fallback
            
            self._create_agent_and_car(agent_id, start_positions, agent_colors, i)

    def _create_agent_and_car(self, agent_id, start_positions, agent_colors, index):
        """Create a single agent and its corresponding car."""
        agent_info = self.config['agents'][agent_id]
        start_idx = agent_info['start_pos_idx'] % len(start_positions)
        initial_state = start_positions[start_idx]
        car_color = agent_colors[index % len(agent_colors)]
        
        # Create car
        self.cars[agent_id] = Car(agent_id, initial_state, self.config['car'], color=car_color)
        
        # Create agent
        self._create_agent(agent_id, agent_info)

    def _create_agent(self, agent_id, agent_info):
        """Create an agent of the specified type."""
        agent_type = agent_info.get('type', 'random') # Default to random if type missing
        
        # Ensure agent_info has necessary fields, provide defaults if missing
        if 'model_path' not in agent_info and agent_type == 'rl':
            print(f"Warning: model_path not specified for RL agent {agent_id}. Agent may not function.")
            agent_info['model_path'] = None # Set to None explicitly
        
        print(f"Creating agent {agent_id} of type {agent_type}")

        if agent_type == 'human':
            self.agents[agent_id] = HumanAgent(agent_id)
        elif agent_type == 'rl':
            # Pass the whole agent_info dict, RL agent can extract what it needs
            self.agents[agent_id] = RLAgent(agent_id, agent_info)
        elif agent_type == 'mpc':
            mpc_specific_params = agent_info # agent_info itself contains horizon, Q_pos etc.
            mpc_specific_params['dt'] = self.dt # Inject dt
            car_cfg_for_mpc = self.config.get('car', {})
            track_cfg_for_mpc = self.config.get('track', {})
            # Ensure all necessary mpc params for the new MPCAgent are in mpc_specific_params
            # (e.g. Q_pos, Q_head etc. should be in config under agent_X for MPC)
            self.agents[agent_id] = MPCAgent(agent_id, mpc_specific_params, car_cfg_for_mpc, track_cfg_for_mpc)
        elif agent_type == 'random':
            # Pass the specific action space for this agent, not the whole dict
            if self.mode == 'single':
                agent_action_space = self.action_space
            else: # Multi-agent mode
                # Access the Box space from the Dict space using the agent_id
                if agent_id in self.action_space.spaces:
                     agent_action_space = self.action_space[agent_id]
                else:
                     # Fallback or error if agent_id somehow isn't in the space dict
                     print(f"Error: Action space for {agent_id} not found in multi-agent dict. Cannot create RandomAgent.")
                     # Handle appropriately - maybe raise error or skip agent creation
                     raise ValueError(f"Action space missing for {agent_id}")
            
            self.agents[agent_id] = RandomAgent(agent_id, agent_action_space)
        else:
            raise ValueError(f"Unknown agent type: {agent_type} for agent {agent_id}")

    def _setup_spaces(self):
        """Set up action and observation spaces based on config."""
        car_cfg = self.config['car']
        
        # --- Action Space --- 
        # Action: [throttle_brake, steer]
        # throttle_brake: -1.0 (max brake) to +1.0 (max throttle)
        # steer: -max_steer_angle to +max_steer_angle
        low_action = np.array([-1.0, -car_cfg['max_steer_angle']], dtype=np.float32)
        high_action = np.array([+1.0, car_cfg['max_steer_angle']], dtype=np.float32)
        action_box = spaces.Box(low=low_action, high=high_action, shape=(2,), dtype=np.float32)

        # --- Observation Space (Dictionary) --- 
        obs_space_dict = {}
        # Define bounds for each potential component
        space_definitions = {
            'x': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            'y': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            'v': spaces.Box(0, car_cfg.get('max_speed', np.inf), shape=(1,), dtype=np.float32),
            'theta': spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            'steer_angle': spaces.Box(-car_cfg['max_steer_angle'], car_cfg['max_steer_angle'], shape=(1,), dtype=np.float32),
            'accel': spaces.Box(
                -car_cfg.get('max_brake_force', 5000) / car_cfg.get('mass', 1500), 
                car_cfg.get('max_engine_force', 4500) / car_cfg.get('mass', 1500), 
                shape=(1,), dtype=np.float32
            ),
            'accel_lat': spaces.Box(-20, 20, shape=(1,), dtype=np.float32), # Approx lateral accel limits
            'dist_to_centerline': spaces.Box(-self.track.width, self.track.width, shape=(1,), dtype=np.float32),
            'dist_to_boundary': spaces.Box(-self.track.half_width, self.track.half_width, shape=(1,), dtype=np.float32),
            'lidar': spaces.Box(0, self.lidar_max_range, shape=(self.lidar_num_beams,), dtype=np.float32),
        }

        # Populate the dict based on configured components
        for component_name in self.observation_components:
            # Add new 'accel_lat' component if requested
            if component_name == 'accel_lat' and 'accel_lat' not in space_definitions:
                 print(f"Warning: Definition for 'accel_lat' was missing, adding default.")
                 space_definitions['accel_lat'] = spaces.Box(-10, 10, shape=(1,), dtype=np.float32) # Default if needed
                 
            if component_name in space_definitions:
                obs_space_dict[component_name] = space_definitions[component_name]
            else:
                print(f"Warning: Definition for observation component '{component_name}' not found. Skipping.")
        
        # Create the final observation space(s)
        observation_dict_space = spaces.Dict(obs_space_dict)

        if self.mode == 'single':
            self.num_agents = 1 
            self._action_space = action_box
            self._observation_space = observation_dict_space
        else:
            # Determine num_agents for multi-mode (needed for Dict keys)
            self.num_agents = self.config.get('simulation', {}).get('num_agents', len(self.config.get('agents', {})))
            if self.num_agents == 0:
                raise ValueError("Multi-agent mode selected but no agents defined in config or num_agents is 0.")
            
            agent_keys = [f'agent_{i}' for i in range(self.num_agents)]
            self._action_spaces = spaces.Dict({agent_id: action_box for agent_id in agent_keys})
            self._observation_spaces = spaces.Dict({agent_id: observation_dict_space for agent_id in agent_keys})

    @property
    def action_space(self):
        if self.mode == 'single':
            return self._action_space
        return self._action_spaces

    @property
    def observation_space(self):
        if self.mode == 'single':
            return self._observation_space
        return self._observation_spaces

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    def _calculate_lidar(self, agent_id, all_car_data):
        num_beams = self.lidar_num_beams
        max_range = self.lidar_max_range
        eps = self.lidar_eps
        x0, y0, theta = all_car_data[agent_id]['x'], all_car_data[agent_id]['y'], all_car_data[agent_id]['theta']
        lidar_distances = np.full(num_beams, max_range, dtype=np.float32)
        angles = theta + np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
        for i, ang in enumerate(angles):
            for r in np.linspace(0, max_range, int(max_range / 100)): # Check every 0.5m for example
                x = x0 + r * np.cos(ang)
                y = y0 + r * np.sin(ang)
                # Check collision with track boundary
                if not self.track.is_on_track(x, y):
                    lidar_distances[i] = r
                    break
                # Check collision with other cars (except self)
                for other_id, other_car in self.cars.items():
                    if other_id == agent_id:
                        continue
                    dx = x - other_car.x
                    dy = y - other_car.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < float(other_car.collision_radius) + eps:
                        lidar_distances[i] = r
                        break
                if lidar_distances[i] < r:
                    break
        return torch.tensor(lidar_distances, dtype=torch.float32, device=device)

    def _get_obs(self):
        """Get observation dictionary for all agents based on configured components."""
        observations = {}
        all_car_data = {agent_id: car.get_data() for agent_id, car in self.cars.items()}
        
        for agent_id, car_data in all_car_data.items():
            agent_instance = self.agents[agent_id]
            obs_dict = {}

            if isinstance(agent_instance, MPCAgent):
                # MPC gets specific observations as numpy arrays or floats
                obs_dict['x'] = float(car_data['x'])
                obs_dict['y'] = float(car_data['y'])
                obs_dict['v'] = float(car_data['v'])
                obs_dict['theta'] = float(car_data['theta'])
                obs_dict['steer_angle'] = float(car_data['steer_angle'])

                if isinstance(self.track, OvalTrack) and self._discretized_centerline_cache:
                    s_curr, _, _, _, _ = self._project_to_oval_centerline(
                        obs_dict['x'], obs_dict['y']
                    )
                    ref_path = self._generate_reference_path(
                        s_current=s_curr,
                        current_car_speed=obs_dict['v'],
                        mpc_agent=agent_instance,
                        car_config=self.config['car']
                    )
                    obs_dict['reference_path'] = ref_path # This is a numpy array
                else:
                    # Fallback: Provide a dummy reference path if track type is not OvalTrack
                    # or centerline cache is not available. MPC will likely fail or perform poorly.
                    print(f"Warning: Could not generate reference path for MPC agent {agent_id}.")
                    dummy_ref_path = np.zeros((agent_instance.N, 4))
                    # Populate with current pos and zero speed to avoid crashes, but MPC will struggle
                    for k_dum in range(agent_instance.N):
                        dummy_ref_path[k_dum,0] = obs_dict['x']
                        dummy_ref_path[k_dum,1] = obs_dict['y']
                        dummy_ref_path[k_dum,2] = obs_dict['theta']
                        dummy_ref_path[k_dum,3] = 0.0 # Target zero speed
                    obs_dict['reference_path'] = dummy_ref_path
            else:
                for component in self.observation_components:
                    if component in car_data:
                        # Directly get from car data if available
                        obs_dict[component] = torch.tensor([car_data[component]], dtype=torch.float32, device=device)
                    elif component == 'dist_to_centerline':
                        # Calculate based on track and car position
                        dist = self.track.get_distance_to_centerline(car_data['x'], car_data['y'])
                        obs_dict[component] = torch.tensor([dist], dtype=torch.float32, device=device)
                    elif component == 'dist_to_boundary':
                        # Calculate distance to centerline first
                        dist_center = self.track.get_distance_to_centerline(car_data['x'], car_data['y'])
                        # Distance to boundary is half_width - dist_center (can be negative if off-track)
                        dist_boundary = self.track.half_width - abs(dist_center)
                        obs_dict[component] = torch.tensor([dist_boundary], dtype=torch.float32, device=device)
                    # Add elif blocks here for other complex/derived observations
                    elif component == 'lidar': 
                        obs_dict[component] = self._calculate_lidar(agent_id, all_car_data)
                    else:
                        # This shouldn't happen if _setup_spaces is correct, but good failsafe
                        print(f"Warning: Cannot get observation component '{component}' for agent {agent_id}.")
                        # Optionally add a default value like np.array([0.0], dtype=np.float32)
            
            observations[agent_id] = obs_dict
            
        return observations

    def _get_info(self):
        """Get info dictionary for all agents."""
        return {
            agent_id: {
                'lap': car.lap_count,
                'total_progress': car.total_progress,
                'speed': car.v,
                'collision': False,
                'off_track': False,
                'lap_completed': False
            }
            for agent_id, car in self.cars.items()
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self._reset_cars()
        self.quit_requested = False # Reset quit flag
        observations = self._get_obs()
        infos = self._get_info()

        if self.render_mode == "human":
            self.render()

        if self.mode == 'single':
            return observations['agent_0'], infos.get('agent_0', {})
        return observations, infos

    def _reset_cars(self):
        """Reset all cars to their starting positions."""
        start_positions = self.track.get_starting_positions(self.num_agents)
        for agent_id in self.agents.keys():
            start_idx = self.config['agents'][agent_id]['start_pos_idx'] % len(start_positions)
            self.cars[agent_id].set_state(start_positions[start_idx])
            self.cars[agent_id].lap_count = 0
            self.cars[agent_id].last_x = self.cars[agent_id].x

    def step(self, actions):
        """Execute one step in the environment."""
        self.current_step += 1
        rewards, terminations, truncations, infos = self._initialize_step()
        
        # Update car states
        delta_progresses =  self._update_car_states(actions)
        
        # Check for collisions and off-track
        collided_agents, off_track_agents = self._check_collisions_and_track()
        
        # Calculate rewards and update termination conditions
        self._calculate_rewards_and_terminations(
            collided_agents, off_track_agents, rewards, terminations, truncations, infos, delta_progresses
        )
        
        observations = self._get_obs()

        if self.render_mode == "human":
            self.render()
            
            # Check if user requested quit during render
            if self.quit_requested:
                print("User requested quit. Ending episode.")
                for agent_id in self.agents.keys():
                    truncations[agent_id] = True # Ensure episode ends
                    infos[agent_id]['quit_signal'] = True # Signal quit

        if self.mode == 'single':
            # Ensure 'quit_signal' is in info even if not quitting
            agent_info = infos.get('agent_0', {})
            agent_info.setdefault('quit_signal', False) 
            return (observations['agent_0'], rewards['agent_0'], 
                   terminations['agent_0'], truncations['agent_0'], 
                   agent_info)
        
        # Ensure 'quit_signal' is in info for multi-agent mode
        for agent_id in self.agents.keys():
            infos[agent_id].setdefault('quit_signal', False)
        return observations, rewards, terminations, truncations, infos

    def _initialize_step(self):
        """Initialize step variables."""
        rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        terminations = {agent_id: False for agent_id in self.agents.keys()}
        truncations = {agent_id: False for agent_id in self.agents.keys()}
        infos = self._get_info()
        return rewards, terminations, truncations, infos

    def _update_car_states(self, actions):
        """Update the state of all cars based on their actions."""
        delta_progresses = {}
        for agent_id, car in self.cars.items():
            action = actions if self.mode == 'single' else actions[agent_id]
            car.update(action, self.dt)
            delta_progress = self.track.calculate_progress(car.last_x, car.last_y, car.x, car.y)
            car.total_progress += delta_progress
            delta_progresses[agent_id] = delta_progress
        return delta_progresses

    def _check_collisions_and_track(self):
        """Check for collisions between cars and off-track conditions."""
        collided_agents = set()
        off_track_agents = set()
        agent_ids = list(self.agents.keys())

        for i in range(len(agent_ids)):
            agent_id_i = agent_ids[i]
            car_i = self.cars[agent_id_i]

            if not self.track.is_on_track(car_i.x, car_i.y):
                off_track_agents.add(agent_id_i)

            for j in range(i + 1, len(agent_ids)):
                agent_id_j = agent_ids[j]
                car_j = self.cars[agent_id_j]
                if check_collision(car_i, car_j):
                    collided_agents.add(agent_id_i)
                    collided_agents.add(agent_id_j)

        return collided_agents, off_track_agents

    def _calculate_rewards_and_terminations(self, collided_agents, off_track_agents, rewards, terminations, truncations, infos, delta_progresseses):
        """Calculate rewards and update termination conditions with improved scaling and shaping for long episodes."""
        # Scale base metrics by lap_bonus so shaping sums to ~one lap's value
        lap_bonus = 400.0
        # Terminal penalties set to one lap penalty
        collision_penalty = -lap_bonus
        off_track_penalty = -lap_bonus
        wrong_direction_penalty = -lap_bonus

        # Intermediate milestone every 10% of lap gives fraction of lap_bonus
        milestone_bonus = lap_bonus * 0.1  # 40
        milestones = set(range(1, 10))      # 10%,20%,...,90%

        # Shaping weights
        k_v = 0.25     # speed deviation weight
        k_c = 0.3      # centerline deviation weight
        k_o = 1.0      # outside-track weight
        # Progress multiplier equals lap_bonus so sum of progress ~ lap_bonus per lap
        k_p = lap_bonus

        car_cfg = self.config['car']
        v_max = 0.75 * car_cfg.get('max_speed', 20.0)
        w_r = self.track.half_width
        max_center_dev = 2 * w_r

        for agent_id, car in self.cars.items():
            infos[agent_id]['lap'] = car.lap_count

            # Basic shaping
            v = car.v
            raw_d_center = self.track.get_distance_to_centerline(car.x, car.y)
            d_center = max(min(raw_d_center, max_center_dev), -max_center_dev)
            outside_dev = max(abs(d_center) - w_r, 0.0)
            d_prog = delta_progresseses.get(agent_id, 0.0)

            reward = (
                - k_v * abs(v - v_max)
                - k_c * abs(d_center)
                - k_o * outside_dev
                + k_p * d_prog
            )

            # Milestone bonus based on fractional part of total_progress
            total_prog = infos[agent_id].get('total_progress', 0.0)
            frac_prog = total_prog - math.floor(total_prog)  # handles >1 correctly
            ms_index = int(frac_prog * 10)
            last_ms = infos[agent_id].get('last_milestone', 0)
            if ms_index in milestones and ms_index > last_ms:
                reward += milestone_bonus
                infos[agent_id]['last_milestone'] = ms_index

            # Termination checks
            if not self.track.is_correct_direction(car.x, car.y, car.theta):
                reward += wrong_direction_penalty
                infos[agent_id]['wrong_direction'] = True
                terminations[agent_id] = True
            else:
                infos[agent_id]['wrong_direction'] = False

            if agent_id in collided_agents:
                reward += collision_penalty
                infos[agent_id]['collision'] = True
                terminations[agent_id] = True

            if agent_id in off_track_agents:
                reward += off_track_penalty
                infos[agent_id]['off_track'] = True
                terminations[agent_id] = True

            # Lap completion
            if self.track.check_lap_completion(car.x, car.last_x, car.y):
                car.lap_count += 1
                reward += lap_bonus
                infos[agent_id]['lap'] = car.lap_count
                # reset milestone counter for new lap
                infos[agent_id]['last_milestone'] = 0

            # Assign
            rewards[agent_id] = reward

            # Truncation
            if self.current_step >= self.max_steps:
                truncations[agent_id] = True

    def render(self):
        """Render the environment and handle Pygame events."""
        if self.ui is not None:
            # Process Pygame events to keep window responsive and detect quit
            if self.render_mode == 'human':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit_requested = True
                        return # Stop rendering if quit is requested
                # pygame.event.pump() # Alternative: just pump events without checking QUIT here
            
            # If still running (no quit), render the UI
            if not self.quit_requested:
                self.ui.render(self.cars, self.current_step, self.dt)

    def close(self):
        """Close the environment and cleanup."""
        if self.ui is not None:
            self.ui.close()
