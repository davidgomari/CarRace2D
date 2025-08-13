# CarRace2D Documentation

This document provides comprehensive information about the CarRace2D simulation environment, including configuration options, agent types, customization guides, and advanced features introduced in version 2.5.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Configuration Guide](#configuration-guide)
4. [Agent Types](#agent-types)
5. [Training Systems](#training-systems)
6. [LiDAR Sensor System](#lidar-sensor-system)
7. [Customization Guide](#customization-guide)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)
11. [Changelog](#changelog)

## Getting Started

### Basic Example

Here's a simple example of running a single-agent simulation with a human-controlled car:

1. Launch the application:
   ```bash
   python main.py
   ```

2. Select "Simulation" mode from the main menu
3. Choose "Single Agent" configuration
4. Select "Human" as the agent type
5. Use arrow keys to control the car:
   - Up/Down: Throttle/Brake
   - Left/Right: Steering

### First Training Run (Single Agent)

To train a basic RL agent in single-agent mode:

1. Configure training parameters in `config_single_agent.yaml`
2. Launch the application
3. Select "Training" mode
4. Choose "Single Agent" configuration
5. Start training and monitor progress
6. Use the UI controls to manage the training process

### Multi-Agent Training

To train multiple RL agents simultaneously:

1. Configure agent and training parameters in `config.yaml` (see the Configuration section below)
2. Launch the application
3. Select "Training" mode
4. Choose "Multi Agent Training" from the menu
5. Training will proceed with each RL agent having its own model and optimizer
6. Use the UI controls to manage the training process

### Headless Training (Cloud Environments)

For training in cloud environments like Kaggle:

```bash
# Single agent training
python train_single_agent.py --config config_kaggle_single.yaml --episodes 200

# Multi-agent training
python train_multi_agent.py --config config_kaggle_multi.yaml --agent-type rl --episodes 150

# Test trained models
python test_trained_model.py --mode single --episodes 5
```

**📖 See [KAGGLE_TRAINING_README.md](KAGGLE_TRAINING_README.md) for detailed cloud training instructions.**

## Project Structure

```
.                     # Project Root
├── agents/           # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py   # Abstract base class for agents
│   ├── human_agent.py  # Keyboard controlled agent
│   ├── mpc_agent.py    # Model Predictive Control agent (using CasADi)
│   ├── random_agent.py # Agent taking random actions
│   └── rl_agent.py     # Reinforcement Learning agent
├── images/           # Contains images like the menu background
│   └── main_menu_background.png
├── models/           # Directory for saving/loading trained models
├── rl_algo/          # Reinforcement Learning algorithms
│   ├── __init__.py
│   └── reinforce.py  # REINFORCE algorithm implementation
├── config.yaml       # Configuration for multi-agent mode
├── config_single_agent.yaml # Configuration for single-agent mode
├── config_kaggle_single.yaml # Kaggle-optimized single agent config
├── config_kaggle_multi.yaml # Kaggle-optimized multi-agent config
├── train_single_agent.py # Headless single agent training script
├── train_multi_agent.py # Headless multi-agent training script
├── test_trained_model.py # Model testing script
├── test_lidar_debug.py # LiDAR debugging script
├── car.py            # Car physics and state implementation
├── environment.py    # Gymnasium environment class (RacingEnv)
├── main.py           # Main entry point, runs the menu and starts simulation/training
├── menu.py           # Implements the Pygame main menu (MainMenu class)
├── README.md         # Main project documentation
├── KAGGLE_TRAINING_README.md # Cloud training guide
├── requirements.txt  # Python dependencies
├── simulation.py     # Contains the simulation loop logic (run_simulation, run_episode)
├── track.py          # Track geometry and related functions (e.g., collision, lap check)
├── training.py       # Functions for training RL agents
├── ui.py             # Pygame UI rendering for the simulation (UI class)
└── utils.py          # Utility functions (e.g., collision checking)
```

## Configuration Guide

The simulation behavior is primarily controlled through YAML configuration files:

*   `config.yaml`: Used for multi-agent simulations and training
*   `config_single_agent.yaml`: Used for single-agent simulations and training
*   `config_kaggle_single.yaml`: Kaggle-optimized single agent configuration
*   `config_kaggle_multi.yaml`: Kaggle-optimized multi-agent configuration

### Configuration Details

#### Simulation Parameters

```yaml
simulation:
  dt: 0.1              # Simulation time step (seconds)
  num_episodes: 3      # Number of episodes to run
  max_steps: 1000      # Max steps per episode
  mode: 'single'       # 'single' or 'multi'
  num_agents: 1        # Number of agents (relevant for multi-agent mode)
  render_mode: 'human' # 'human', 'rgb_array', or None
  render_fps: 60       # Frames per second for rendering
  print_lidar: false   # Print LiDAR values for debugging
```

*   **dt:** Controls the simulation time step. Smaller values provide more accurate physics but slower simulation.
*   **num_episodes:** Number of episodes to run in simulation mode.
*   **max_steps:** Maximum steps per episode before truncation.
*   **mode:** 'single' for single-agent mode, 'multi' for multi-agent mode.
*   **num_agents:** Number of agents in multi-agent mode.
*   **render_mode:** 
    *   'human': Display the simulation with UI controls
    *   'rgb_array': Return frames as numpy arrays (for recording)
    *   None: No rendering (fastest, for headless operation)
*   **render_fps:** Controls the frame rate when rendering.
*   **print_lidar:** Enable LiDAR debugging output.

#### Training Parameters

```yaml
training:
  num_episodes: 500    # Number of training episodes
  max_steps: 1000      # Max steps per episode
  render_mode: 'human' # 'human', 'rgb_array', or None
  render_fps: 60       # Frames per second for rendering
  save_frequency: 100  # Save model every N episodes
  learning_rate: 0.001 # Learning rate for the optimizer
  discount_factor: 0.99 # Discount factor for the optimizer
  rl_algo: 'reinforce' # 'reinforce', 'ppo', 'td3', etc.
  resume_training: True # If True, resume from last saved model if available
```

*   **num_episodes:** Number of episodes to train for.
*   **max_steps:** Maximum steps per episode before truncation.
*   **render_mode:** 
    *   'human': Display the training with UI controls (useful for monitoring)
    *   'rgb_array': Return frames as numpy arrays (for recording)
    *   None: No rendering (fastest, recommended for training)
*   **render_fps:** Controls the frame rate when rendering.
*   **save_frequency:** Save the model every N episodes.
*   **learning_rate:** Learning rate for the optimizer.
*   **discount_factor:** Discount factor for calculating returns.
*   **rl_algo:** The RL algorithm to use ('reinforce' by default).
*   **resume_training:** If True, training resumes from the last saved model (if available).

> **Note:** For multi-agent RL training, use `config.yaml` and define each agent under the `agents` section. Each RL agent will have its own model and training progress. For faster training, set `render_mode` to `None` in the training configuration.

#### Track Parameters

```yaml
track:
  type: 'oval'         # Track type
  length: 100.0        # Length of straight sections
  radius: 30.0         # Radius of curved sections
  width: 15.0          # Total width of the track
  start_line_x: 0.0    # X-coordinate of the start/end line center
  start_lane: 'bottom' # 'top' or 'bottom'
```

*   **type:** Currently supports 'oval' track type.
*   **length:** Length of the straight sections.
*   **radius:** Radius of the curved sections.
*   **width:** Total width of the track.
*   **start_line_x:** X-coordinate of the start/finish line.
*   **start_lane:** Which lane the start line is on ('top' or 'bottom').

#### Environment Parameters

```yaml
environment:
  observation_components: 
    - 'v'
    - 'steer_angle'
    - 'accel'
    - 'dist_to_centerline'
    - 'lidar'  # Available in multi-agent mode
    # Optionally: 'accel_lat', 'x', 'y', 'theta', etc.
```

*   **observation_components:** List of state components to include in the observation space. Available options include:
    *   'x': X-position
    *   'y': Y-position
    *   'v': Velocity
    *   'theta': Heading angle
    *   'steer_angle': Steering angle
    *   'dist_to_centerline': Distance to track centerline
    *   'accel': Longitudinal acceleration
    *   'accel_lat': Lateral acceleration
    *   'lidar': LiDAR sensor readings (multi-agent only)

#### Car Parameters

```yaml
car:
  wheelbase: 2.5         # L (meters)
  mass: 1500             # Mass (kg)
  max_speed: 20.0        # m/s
  min_accel: -5.0        # m/s^2 (braking)
  max_accel: 3.0         # m/s^2
  max_steer_angle: 0.6   # radians (~34 degrees)
  width: 1.8             # For collision/visualization
  length: 4.0            # For collision/visualization
  collision_radius: 1.5  # Simplified collision check radius
  
  # Physics coefficients
  coeff_drag: 0.8         # Air resistance coefficient
  coeff_rolling_resistance: 60.0 # Rolling resistance coefficient
  coeff_friction: 1.1      # Combined tire friction coefficient
  coeff_cornering_stiffness: 15.0 # Tire lateral stiffness factor
  max_engine_force: 4500   # Max forward force from engine (N)
  max_brake_force: 6000    # Max backward force from brakes (N)
  gravity: 9.81          # Acceleration due to gravity (m/s^2)
  
  # LiDAR sensor configuration (multi-agent mode)
  lidar_num_beams: 24     # Number of LiDAR beams
  lidar_max_range: 50.0   # Maximum LiDAR range
  lidar_eps: 1e-3         # LiDAR precision
```

*   **wheelbase:** Distance between front and rear axles.
*   **mass:** Vehicle mass in kg.
*   **max_speed:** Maximum speed in m/s.
*   **min_accel/max_accel:** Acceleration limits in m/s².
*   **max_steer_angle:** Maximum steering angle in radians.
*   **width/length:** Vehicle dimensions for collision detection and visualization.
*   **collision_radius:** Simplified radius for collision detection.
*   **Physics coefficients:** Various coefficients for the physics model.
*   **LiDAR parameters:** Configuration for the LiDAR sensor system.

#### Agent Configuration

For single-agent mode, agent configuration is in `config_single_agent.yaml`:

```yaml
agent_config:
  human:
    description: "Controlled by user via keyboard arrow keys."
  random:
    description: "Takes random actions within the action space."
  mpc:
    description: "Model Predictive Control agent."
    horizon: 15       # Prediction horizon (N)
    # MPC cost function weights
    Q_progress: 5.0
    Q_pos: 1.0
    Q_head: 0.2
    Q_vel: 5.0
    R_accel: 0.5
    R_steer_cmd: 0.2
    R_steer_rate: 0.05
    # MPC constraints
    mpc_max_accel: 3.0
    mpc_min_accel: -5.0
    max_steer_rate_sim: 3.14
    # Collision avoidance
    max_opponents_to_consider: 0
    default_opponent_collision_radius: 1.5
    # Feature flags
    use_progress_cost: true
    use_path_tracking_cost: false
    use_control_effort_cost: false
    use_heading_cost: true
    use_velocity_cost: true
    use_steer_rate_cost: false
    use_terminal_cost: false
    apply_track_boundaries: true
    apply_collision_avoidance: false
  rl:
    description: "Reinforcement Learning agent."
    model_path: "models/single_agent/trained_model.zip" # Path to the saved RL model
```

For multi-agent mode, agent configuration is in `config.yaml`:

```yaml
agents:
  agent_0:
    type: 'rl'           # rl, mpc, human, random
    model_path: 'models/multi_agent/agent_0_model.zip' # Path for RL model
    start_pos_idx: 0     # Index for starting position
  agent_1:
    type: 'mpc'
    start_pos_idx: 1
  agent_2:
    type: 'random'
    start_pos_idx: 2
```

*   **type:** Agent type ('human', 'random', 'mpc', 'rl').
*   **model_path:** Path to the saved model for RL agents.
*   **start_pos_idx:** Index for the starting position on the track.
*   **horizon:** Prediction horizon for MPC agents.

#### RL Hyperparameters

```yaml
rl:
  learning_rate: 0.0003
  discount_factor: 0.99
  # ... other hyperparameters
```

*   **learning_rate:** Learning rate for RL algorithms.
*   **discount_factor:** Discount factor for calculating returns.

#### UI Sidebar Configuration

```yaml
ui:
  sidebar_columns:
    - { key: 'agent_id', header: 'Agent', format: '' }
    - { key: 'agent_type', header: 'Type', format: '' }
    - { key: 'x', header: 'X', format: '.1f' }
    - { key: 'y', header: 'Y', format: '.1f' }
    - { key: 'v', header: 'Spd', format: '.1f' }
    - { key: 'lap', header: 'Lap', format: 'd' }
    - { key: 'dist_to_centerline', header: 'dC', format: '.2f' }
    - { key: 'dist_to_boundary', header: 'dB', format: '.2f' }
    - { key: 'total_progress', header: 'P%', format: '.2f'}
```

*   **sidebar_columns:** List of columns to display in the sidebar. Each entry specifies the data key, header, and format.

## Agent Types

### Human Agent
- **Control**: Keyboard arrow keys
- **Features**: Real-time human control with visual feedback
- **Use Cases**: Manual testing, baseline comparison, interactive demonstrations

### Random Agent
- **Behavior**: Random actions within action space bounds
- **Features**: Uniform random sampling of acceleration and steering
- **Use Cases**: Baseline performance, testing environment stability

### MPC Agent (Model Predictive Control)
- **Algorithm**: CasADi-based optimization
- **Features**: 
  - Modular cost functions (progress, path tracking, control effort)
  - Collision avoidance with other agents
  - Track boundary constraints
  - Configurable prediction horizon
- **Use Cases**: Optimal control, path following, collision avoidance

### RL Agent (Reinforcement Learning)
- **Algorithm**: REINFORCE (default, extensible)
- **Features**:
  - Policy gradient learning
  - Model saving/loading
  - Configurable observation space
  - GPU acceleration support
- **Use Cases**: Learning-based control, policy optimization, multi-agent learning

## Training Systems

### GUI Training
- **Access**: Through main menu interface
- **Features**: Real-time visualization, interactive controls
- **Best For**: Development, debugging, small-scale experiments

### Headless Training
- **Scripts**: `train_single_agent.py`, `train_multi_agent.py`
- **Features**: No GUI dependency, cloud-ready, command-line interface
- **Best For**: Large-scale training, cloud environments, production runs

### Training Features
- **Resume Training**: Continue from saved checkpoints
- **Model Saving**: Automatic periodic saves
- **Progress Tracking**: Episode rewards, loss values, performance metrics
- **Multi-Agent**: Simultaneous training of multiple RL agents

## LiDAR Sensor System

### Overview
The LiDAR sensor system provides distance measurements to obstacles and track boundaries, enabling advanced perception capabilities for multi-agent scenarios.

### Configuration
```yaml
car:
  lidar_num_beams: 24     # Number of LiDAR beams (angular resolution)
  lidar_max_range: 50.0   # Maximum detection range
  lidar_eps: 1e-3         # Numerical precision
```

### Features
- **Multi-beam Detection**: Configurable number of beams for angular resolution
- **Collision Detection**: Detects other vehicles within range
- **Track Boundary Sensing**: Detects track boundaries and obstacles
- **Vectorized Computation**: Optimized for performance
- **Debug Support**: Built-in debugging tools

### Usage
1. **Enable in Configuration**:
   ```yaml
   environment:
     observation_components:
       - 'lidar'
   ```

2. **Debug LiDAR**:
   ```bash
   python test_lidar_debug.py
   ```

3. **Monitor Output**:
   ```yaml
   simulation:
     print_lidar: true
   ```

### Technical Details
- **Coordinate System**: Polar coordinates around agent position
- **Collision Detection**: Uses vehicle collision radius
- **Track Boundaries**: Ray-casting to track edges
- **Performance**: GPU-accelerated tensor operations

## Customization Guide

### Customizing the UI Sidebar

You can fully customize the columns shown in the sidebar information table via the `ui.sidebar_columns` section in your config file. Each column can display any property from the car state or agent info, and you can specify the header and formatting.

Example:
```yaml
ui:
  sidebar_columns:
    - { key: 'agent_id', header: 'Agent', format: '' }
    - { key: 'agent_type', header: 'Type', format: '' }
    - { key: 'x', header: 'X', format: '.1f' }
    - { key: 'y', header: 'Y', format: '.1f' }
    - { key: 'v', header: 'Spd', format: '.1f' }
    - { key: 'lap', header: 'Lap', format: 'd' }
    - { key: 'dist_to_centerline', header: 'dC', format: '.2f' }
    - { key: 'dist_to_boundary', header: 'dB', format: '.2f' }
    - { key: 'total_progress', header: 'P%', format: '.2f'}
```

If the `ui.sidebar_columns` section is missing, a minimal default sidebar will be shown.

### Modifying Observations

To add or remove observations from the environment:

1. **Update Configuration File:**
   - Open either `config.yaml` or `config_single_agent.yaml`
   - Locate the `environment` section:
   ```yaml
   environment:
     observation_components:
       - 'v'
       - 'steer_angle'
       - 'accel'
       - 'dist_to_centerline'
       - 'lidar'  # Multi-agent only
       # Optionally: 'accel_lat', 'x', 'y', 'theta', etc.
   ```
   - Add or remove components from this list

2. **Define Observation Space:**
   - Open `environment.py`
   - Locate the `_setup_spaces` method in the `RacingEnv` class
   - Add your new observation to the `space_definitions` dictionary:
   ```python
   space_definitions = {
       'x': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
       'y': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
       # Add your new observation here, defining its bounds and shape
       'my_new_obs': spaces.Box(low_value, high_value, shape=(1,), dtype=np.float32),
   }
   ```

3. **Implement Observation Calculation:**
   - In `environment.py`, find the `_get_obs` method
   - Add logic to calculate your new observation:
   ```python
   def _get_obs(self):
       observations = {}
       for agent_id, car_data in all_car_data.items():
           obs_dict = {}
           for component in self.observation_components:
               if component == 'my_new_obs':
                   # Add your calculation logic here
                   obs_dict[component] = np.array([value], dtype=np.float32)
   ```

### Customizing Car Physics

1. **Modify Physical Parameters:**
   - Edit the `car` section in the configuration files:
   ```yaml
   car:
     wheelbase: 2.5         # Car's wheelbase length
     mass: 1500             # Car's mass
     max_speed: 20.0        # Maximum speed
     width: 1.8             # Car's width
     length: 4.0            # Car's length
     
     # Physics coefficients
     coeff_drag: 0.8
     coeff_rolling_resistance: 60.0
     coeff_friction: 1.1
     coeff_cornering_stiffness: 15.0
     max_engine_force: 4500
     max_brake_force: 6000
   ```

2. **Modify Physics Model:**
   - The core physics logic is in `car.py` in the `update` method
   - Adjust force calculations, acceleration models, or add new physics components

### Customizing Track Layout

1. **Basic Track Parameters:**
   - Modify the `track` section in configuration:
   ```yaml
   track:
     type: 'oval'          # Track type
     length: 100.0         # Straight section length
     radius: 30.0          # Curve radius
     width: 15.0           # Track width
     start_line_x: 0.0     # Start line position
     start_lane: 'bottom'  # Start line location
   ```

2. **Advanced Track Customization:**
   - Implement a new track type by:
     1. Create a new track class in `track.py`
     2. Implement required methods: `is_on_track`, `get_distance_to_centerline`, `check_lap_completion`
     3. Add the new track type to the track initialization in `RacingEnv._setup_track`

### Customizing Rewards

1. **Modify Reward Calculation:**
   - In `environment.py`, find the `_calculate_rewards_and_terminations` method
   - Adjust reward factors:
   ```python
   progress_reward_factor = 0.1    # Speed-based progress reward
   collision_penalty = -100.0      # Penalty for collisions
   off_track_penalty = -50.0       # Penalty for going off-track
   lap_bonus = 100.0              # Bonus for completing a lap
   ```
   - Add new reward components based on your requirements

### Using a Different RL Algorithm

The project currently uses the REINFORCE algorithm by default, but you can implement and use different RL algorithms:

1. **Create a New Algorithm:**
   - Add a new file in the `rl_algo/` directory (e.g., `ppo.py`, `td3.py`, etc.)
   - Implement the algorithm class with the following interface:
   ```python
   class MyRLAlgorithm(nn.Module):
       def __init__(self, state_dim, action_dim, **kwargs):
           super().__init__()
           # Initialize your network architecture
           
       def forward(self, x):
           # Implement the forward pass
           # Should return (mean, std) for continuous action spaces
           return mean, std
           
       def act(self, state):
           # Convert state to tensor if needed
           # Get action from the policy
           # Return (action, log_prob)
           return action, log_prob
   ```

2. **Update Configuration:**
   - In `config_single_agent.yaml` or `config.yaml`, set the RL algorithm:
   ```yaml
   training:
     rl_algo: 'my_algorithm'  # Name of your algorithm
     learning_rate: 0.001
     discount_factor: 0.99
     # Add algorithm-specific parameters
   ```

3. **Register the Algorithm:**
   - In `training.py`, update the `single_agent_training` function to handle your algorithm:
   ```python
   if rl_algo == 'reinforce':
       obs_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
       rl_agent.model = ReinforcePolicy(obs_dim, env.action_space.shape[0])
   elif rl_algo == 'my_algorithm':
       obs_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
       rl_agent.model = MyRLAlgorithm(obs_dim, env.action_space.shape[0])
   else:
       raise ValueError(f"Unsupported RL algorithm: {rl_algo}")
   ```

4. **Update Model Loading:**
   - Ensure your algorithm can load saved models by implementing the appropriate methods
   - The `RLAgent` class in `agents/rl_agent.py` handles model loading and inference

### Adding New Agent Types

1. **Create Agent Class:**
   - Add a new file in the `agents/` directory (e.g., `my_agent.py`)
   - Inherit from `BaseAgent` class
   - Implement the `get_action` method

2. **Register Agent:**
   - Update `environment.py`'s `_create_agent` method to include your new agent type
   - Add agent-specific configuration in `config_single_agent.yaml`:
   ```yaml
   agent_config:
     my_agent:
       description: "My custom agent"
       # Add agent-specific parameters
   ```

### Visualization Customization

1. **Modify UI Elements:**
   - Adjust the UI layout in `ui.py`
   - Customize colors, dimensions, and information display in the `_draw_sidebar` method
   - Modify car visualization in `_draw_cars` method

2. **Add New Visualizations:**
   - Add new visualization methods in the `UI` class
   - Call them from the `_render_human` method

## Performance Optimization

### Simulation Performance

1. **Rendering Optimization:**
   - Set `render_mode: None` for headless operation
   - Reduce `render_fps` for better performance
   - Disable unnecessary visual elements

2. **Physics Optimization:**
   - Adjust `dt` in simulation parameters
   - Use simplified collision detection for large numbers of agents
   - Optimize track complexity

3. **Memory Management:**
   - Clear unused models and resources
   - Monitor memory usage during long training sessions
   - Use appropriate batch sizes for RL training

### Training Performance

1. **RL Training Optimization:**
   - Use appropriate batch sizes
   - Implement early stopping
   - Save checkpoints regularly
   - Monitor training metrics

2. **Hardware Utilization:**
   - Enable GPU acceleration where available
   - Optimize CPU usage for physics calculations
   - Balance rendering and computation

### Cloud Environment Optimization

1. **Kaggle-Specific Tips:**
   - Use Kaggle-optimized configuration files
   - Start with fewer episodes for initial testing
   - Monitor GPU usage with `nvidia-smi`
   - Save models frequently

2. **Headless Training:**
   - Use dedicated training scripts
   - Set `render_mode: None`
   - Use conservative learning rates
   - Implement proper error handling

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Ensure the config file exists in the current directory
   - Use `--config` to specify the correct path

2. **Model loading errors**
   - Check that model files exist when using `--resume`
   - Ensure model paths in config files are correct

3. **Training too slow**
   - Use Kaggle-optimized config files
   - Reduce episodes and max steps
   - Use `render_mode: None` in training section

4. **Memory issues**
   - Reduce batch sizes or episode lengths
   - Use simpler observation spaces
   - Reduce LiDAR beam count

5. **LiDAR Issues**
   - Use `test_lidar_debug.py` script to verify calculations
   - Check LiDAR configuration parameters
   - Ensure LiDAR is enabled in observation components

6. **MPC Convergence Issues**
   - Adjust solver parameters in configuration
   - Check cost function weights
   - Verify constraint configurations

### Performance Tips

1. **Use Kaggle-optimized configs** - They're designed for faster training
2. **Start with fewer episodes** - 100-500 episodes for initial testing
3. **Monitor GPU usage** - Use `nvidia-smi` if available
4. **Save frequently** - Use `--save-frequency 50` or lower
5. **Use conservative learning rates** - 0.001 or lower for stability

### Debug Tools

1. **LiDAR Debugging**:
   ```bash
   python test_lidar_debug.py
   ```

2. **Model Testing**:
   ```bash
   python test_trained_model.py --mode single --episodes 5
   ```

3. **Configuration Validation**:
   - Check YAML syntax
   - Verify file paths
   - Validate parameter ranges

## API Reference

### Environment Class (RacingEnv)

#### Key Methods

- `reset()`: Reset environment to initial state
- `step(actions)`: Execute actions and return new state
- `render()`: Render current state
- `close()`: Clean up resources

#### Key Attributes

- `observation_space`: Gymnasium observation space
- `action_space`: Gymnasium action space
- `cars`: Dictionary of car objects
- `agents`: Dictionary of agent objects
- `track`: Track object

### Agent Base Class

#### Required Methods

- `get_action(observation)`: Return action for given observation
- `reset()`: Reset agent state

### Car Class

#### Key Methods

- `update(action, dt)`: Update car state based on action
- `get_data()`: Return current car state data
- `reset()`: Reset car to initial state

#### Key Attributes

- `x, y`: Position coordinates
- `v`: Velocity
- `theta`: Heading angle
- `steer_angle`: Steering angle

## FAQ

### General Questions

**Q: Can I use my own custom track?**  
A: Yes, you can create custom tracks by implementing a new track class following the existing track interface. Also, you need to change the track draw method inside the `ui.py`.

**Q: How do I save and load trained models?**  
A: Models are automatically saved during training. You can load them by specifying the model path in the configuration file.

**Q: Can I use multiple RL algorithms?**  
A: Yes, the system supports different RL algorithms. You can implement new algorithms by following the existing interface.

**Q: How do I train in cloud environments?**  
A: Use the dedicated headless training scripts (`train_single_agent.py`, `train_multi_agent.py`) with Kaggle-optimized configurations.

### Technical Questions

**Q: How do I modify the physics model?**  
A: The physics model is implemented in `car.py`. You can modify the `update` method to change the physics behavior.

**Q: How do I add new observation components?**  
A: Add new components to the observation space in `environment.py` and implement their calculation in the `_get_obs` method.

**Q: How do I implement a new agent type?**  
A: Create a new agent class inheriting from `BaseAgent` and implement the required methods.

**Q: How do I configure LiDAR sensors?**  
A: Enable LiDAR in observation components and configure parameters in the car section of your config file.

## Known Issues

1. **Physics Limitations:**
   - Simplified collision detection may miss some edge cases
   - Tire model is basic and may not reflect real-world behavior
   - Some physics parameters may need tuning for specific scenarios

2. **RL Training:**
   - Some RL algorithms may require additional hyperparameter tuning
   - Training stability can vary with different configurations

3. **UI/Visualization:**
   - High agent counts may impact performance
   - Some visual elements may not scale well with window size
   - Limited support for custom visualizations

4. **MPC Performance:**
   - Solver convergence may vary with different configurations
   - Large prediction horizons may impact performance
   - Collision avoidance may not work perfectly in all scenarios

## Changelog

### Version 2.5 (Current)
- **Headless Training Support:** Added dedicated training scripts for Kaggle and cloud environments
- **Enhanced LiDAR System:** Configurable LiDAR sensor with collision detection and track boundary sensing
- **Improved MPC Agent:** Modular cost functions, collision avoidance, and enhanced track boundary constraints
- **Kaggle-Optimized Configurations:** Pre-configured settings for faster training in cloud environments
- **Debug Tools:** Added LiDAR debugging script and enhanced error handling
- **Documentation Updates:** Comprehensive guide for headless training and new features

### Version 2.4
- Improving MPC in multi-agent environment but still not working

### Version 2.3
- Fixing MPC agent for single agent mode

### Version 2.2
- Multi-agent RL training support

### Version 2.1
- Resume training from saved models (`resume_training` option)
- Improved reward shaping and progress/lap tracking
- Added support for new observation components (`accel`, `accel_lat`)
- Improved car physics with friction circle and lateral acceleration
- Performance optimizations and UI enhancements
- Training automatically uses GPU if available

### Version 2.0
- Updated UI with a cyberpunk modern city theme
- Added Pause/Resume and Reset buttons to the UI
- Improved vehicle visualization
- Integrated single-agent RL training with configurable algorithms (default: REINFORCE)
- Simulation now correctly loads the trained model from the path specified in the config file
- Fixed bugs related to loading/saving PyTorch RL models
- Minor bug fixes and code cleanup

### Version 1.0
- Initial release
- Basic car physics
- Simple track implementation
- Human, Random, MPC agent support
- Added configuration file support
- Basic UI implementation 
- Simulation environment
- Support for multiple agent types

