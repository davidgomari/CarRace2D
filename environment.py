# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pygame # For rendering and human input
import yaml
import os # Added import os

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
        self._setup_spaces()
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
        self.num_agents = sim_cfg['num_agents']

        # Load observation components config
        env_cfg = self.config.get('environment', {})
        self.observation_components = env_cfg.get('observation_components', ['x', 'y', 'v', 'theta'])
        print(f"Using observation components: {self.observation_components}")

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
            # MPC agent expects specific nested params
            mpc_params = agent_info # Top-level params like horizon are directly in agent_info now
            mpc_params['dt'] = self.dt # Inject dt
            car_cfg = self.config.get('car', {})
            track_cfg = self.config.get('track', {})
            self.agents[agent_id] = MPCAgent(agent_id, mpc_params, car_cfg, track_cfg)
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
            # Use actual forces for accel bounds if available, else legacy
            'accel': spaces.Box(
                -car_cfg.get('max_brake_force', 5000) / car_cfg.get('mass', 1500), 
                car_cfg.get('max_engine_force', 4500) / car_cfg.get('mass', 1500), 
                shape=(1,), dtype=np.float32
            ),
            'accel_lat': spaces.Box(-10, 10, shape=(1,), dtype=np.float32), # Approx lateral accel limits
            'dist_to_centerline': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            # Add definitions for other potential future obs
            'dist_to_boundary': spaces.Box(-np.inf, self.track.half_width, shape=(1,), dtype=np.float32),
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

    def _get_obs(self):
        """Get observation dictionary for all agents based on configured components."""
        observations = {}
        all_car_data = {agent_id: car.get_data() for agent_id, car in self.cars.items()}
        
        for agent_id, car_data in all_car_data.items():
            obs_dict = {}
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
                # elif component == 'lidar': 
                #     obs_dict[component] = self._calculate_lidar(agent_id, all_car_data)
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
        """Calculate rewards and update termination conditions."""
        # Penalties for terminal states - should be significantly negative
        collision_penalty = -10000.0 # Increased penalty (was -20.0)
        off_track_penalty = -10000.0 # Added explicit penalty
        wrong_direction_penalty = -10000.0 # Added penalty, potentially worse than off-track/collision
        lap_bonus = 400.0          # Optional bonus for completing a lap

        # --- Reward shaping parameters --- 
        k_1 = 0.25   # Speed deviation
        k_2 = 0.3   # Distance to centerline
        k_3 = 1.0   # Being outside track width
        k_4 = 50000.0 # Progress multiplier
        car_cfg = self.config['car']
        v_max = 0.75 * car_cfg.get('max_speed', 20.0)
        w_r = self.track.half_width

        for agent_id, car in self.cars.items():
            # Update lap count info
            infos[agent_id]['lap'] = car.lap_count

            # --- Base Reward Calculation (Shaping) ---
            v = car.v
            d_center = self.track.get_distance_to_centerline(car.x, car.y)
            d_prog = delta_progresseses.get(agent_id, 0.0) # Progress increment
            
            reward = (
                - k_1 * abs(v - v_max)       # Penalty for speed deviation
                - k_2 * abs(d_center)        # Penalty for centerline deviation
                - k_3 * max(abs(d_center) - w_r, 0) # Additional Penalty for being outside track width
                + k_4 * d_prog               # Reward for progress
            )

            # --- Penalties & Termination for Bad States --- 
            
            # Check direction
            if not self.track.is_correct_direction(car.x, car.y, car.theta):
                reward += wrong_direction_penalty
                # terminations[agent_id] = True
                infos[agent_id]['wrong_direction'] = True
            else:
                infos[agent_id]['wrong_direction'] = False

            # Check collision
            if agent_id in collided_agents:
                reward += collision_penalty
                # terminations[agent_id] = True
                infos[agent_id]['collision'] = True

            # Check off-track
            if agent_id in off_track_agents:
                reward += off_track_penalty
                # terminations[agent_id] = True
                infos[agent_id]['off_track'] = True

            # --- Lap Completion Bonus --- 
            lap_completed = self.track.check_lap_completion(car.x, car.last_x, car.y)
            infos[agent_id]['lap_completed'] = lap_completed
            if lap_completed:
                car.lap_count += 1
                reward += lap_bonus # Add bonus for completing the lap
                infos[agent_id]['lap'] = car.lap_count

            # Store final reward for the step
            rewards[agent_id] = reward

            # Check truncation (max steps)
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
