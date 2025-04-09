# CarRace2D Documentation

This document provides detailed information about the CarRace2D simulation environment, including configuration options, agent types, and customization guides.

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

### First Training Run

To train a basic RL agent:

1. Configure training parameters in `config_single_agent.yaml`
2. Launch the application
3. Select "Training" mode
4. Choose "Single Agent" configuration
5. Start training and monitor progress
6. Use the UI controls to manage the training process

## Notes

*   The Reinforcement Learning implementation now supports model loading and inference, with a configurable algorithm selection.
*   The car physics model is a simplified kinematic bicycle model with added force dynamics. It may not perfectly reflect real-world vehicle behavior.
*   The UI now includes interactive controls (Back, Reset, Pause/Resume) that work during both simulation and training.
*   Multi-agent RL training is not yet implemented. The multi-agent training function is a placeholder for future development.

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
├── car.py            # Car physics and state implementation
├── environment.py    # Gymnasium environment class (RacingEnv)
├── main.py           # Main entry point, runs the menu and starts simulation/training
├── menu.py           # Implements the Pygame main menu (MainMenu class)
├── README.md         # This file
├── requirements.txt  # Python dependencies
├── simulation.py     # Contains the simulation loop logic (run_simulation, run_episode)
├── track.py          # Track geometry and related functions (e.g., collision, lap check)
├── training.py       # Functions for training RL agents
├── ui.py             # Pygame UI rendering for the simulation (UI class)
└── utils.py          # Utility functions (e.g., collision checking)
```

## Customization Guide

### Modifying Observations

To add or remove observations from the environment:

1. **Update Configuration File:**
   - Open either `config.yaml` or `config_single_agent.yaml`
   - Locate the `environment` section:
   ```yaml
   environment:
     observation_components:
       - 'x'
       - 'y'
       - 'v'
       - 'theta'
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

### Performance Optimization

1. **Adjust Simulation Parameters:**
   ```yaml
   simulation:
     dt: 0.1              # Time step (smaller = more accurate but slower)
     render_fps: 60       # Frame rate for visualization
     max_steps: 1000      # Maximum steps per episode
   ```

2. **Modify Physics Update Frequency:**
   - Adjust the physics calculations in `car.py`
   - Consider implementing sub-stepping for more accurate physics at lower update rates

Remember to test thoroughly after making any modifications, as changes to one component may affect other parts of the simulation.

## Configuration

The simulation behavior is primarily controlled through two YAML configuration files:

*   `config.yaml`: Used for multi-agent simulations and training.
*   `config_single_agent.yaml`: Used for single-agent simulations and training. It also contains the `agent_config` section detailing parameters for different agent types when run in single mode.

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

> **Note:** For faster training, set `render_mode` to `None` in the training configuration. This will run the training without GUI visualization, which is significantly faster. Use `render_mode: 'human'` when you want to monitor the training progress visually.

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
    - 'x'
    - 'y'
    - 'v'
    - 'theta'
    - 'steer_angle'
    - 'dist_to_centerline'
```

*   **observation_components:** List of state components to include in the observation space. Available options include:
    *   'x': X-position
    *   'y': Y-position
    *   'v': Velocity
    *   'theta': Heading angle
    *   'steer_angle': Steering angle
    *   'dist_to_centerline': Distance to track centerline

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
```

*   **wheelbase:** Distance between front and rear axles.
*   **mass:** Vehicle mass in kg.
*   **max_speed:** Maximum speed in m/s.
*   **min_accel/max_accel:** Acceleration limits in m/s².
*   **max_steer_angle:** Maximum steering angle in radians.
*   **width/length:** Vehicle dimensions for collision detection and visualization.
*   **collision_radius:** Simplified radius for collision detection.
*   **Physics coefficients:** Various coefficients for the physics model.

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
  rl:
    description: "Reinforcement Learning agent."
    model_path: "models/single_agent/trained_model.zip" # Path to the saved RL model
```

For multi-agent mode, agent configuration is in `config.yaml`:

```yaml
agents:
  agent_0:
    type: 'rl'           # rl, mpc, human, random
    model_path: 'models/agent0_policy.zip' # Path for RL model
    start_pos_idx: 0     # Index for starting position
  agent_1:
    type: 'human'
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

## Performance Tips

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

## FAQ

### General Questions

**Q: Can I use my own custom track?**  
A: Yes, you can create custom tracks by implementing a new track class following the existing track interface. Also, you need to change the track draw method inside the `ui.py`.

**Q: How do I save and load trained models?**  
A: Models are automatically saved during training. You can load them by specifying the model path in the configuration file.

**Q: Can I use multiple RL algorithms?**  
A: Yes, the system supports different RL algorithms. You can implement new algorithms by following the existing interface.

### Technical Questions

**Q: How do I modify the physics model?**  
A: The physics model is implemented in `car.py`. You can modify the `update` method to change the physics behavior.

**Q: How do I add new observation components?**  
A: Add new components to the observation space in `environment.py` and implement their calculation in the `_get_obs` method.

**Q: How do I implement a new agent type?**  
A: Create a new agent class inheriting from `BaseAgent` and implement the required methods.

## Known Issues

1. **Physics Limitations:**
   - Simplified collision detection may miss some edge cases
   - Tire model is basic and may not reflect real-world behavior
   - Some physics parameters may need tuning for specific scenarios

2. **RL Training:**
   - Multi-agent RL training is not yet implemented
   - Some RL algorithms may require additional hyperparameter tuning
   - Training stability can vary with different configurations

3. **UI/Visualization:**
   - High agent counts may impact performance
   - Some visual elements may not scale well with window size
   - Limited support for custom visualizations

## Changelog

### Version 2.0
- Updated UI with a cyberpunk modern city theme.
- Added Pause/Resume and Reset buttons to the UI.
- Improved vehicle visualization.
- Integrated single-agent RL training with configurable algorithms (default: REINFORCE).
- Simulation now correctly loads the trained model from the path specified in the config file.
- Fixed bugs related to loading/saving PyTorch RL models.
- Minor bug fixes and code cleanup.

### Version 1.0
- Initial release
- Basic car physics
- Simple track implementation
- Human, Random, MPC agent support
- Added configuration file support
- Basic UI implementation 
- Simulation environment
- Support for multiple agent types
