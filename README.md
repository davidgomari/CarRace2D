# CarRace2D Simulation

A 2D racing simulation environment built with Python, Pygame, and Gymnasium. It supports single and multi-agent simulations with various agent types (Human, RL, MPC, Random) and features a graphical user interface.

## Features

*   **Graphical User Interface:** Main menu for selecting modes (Simulation/Training) and agent configurations using Pygame.
*   **Simulation Modes:** Supports both single-agent and multi-agent simulations.
*   **Agent Types:** Includes Human (keyboard control), Random, basic Model Predictive Control (MPC using CasADi), and a placeholder for Reinforcement Learning (RL) agents.
*   **Configurable Environment:** Simulation parameters, track layout, car physics, agent types, and more can be configured via YAML files.
*   **Modular Structure:** Code is organized into modules for UI, environment, track, car, agents, etc.
*   **Basic Physics:** Implements a kinematic bicycle model with considerations for engine/brake forces, drag, and rolling resistance.

## Visuals

![Simulation - Multi Agent Mode](/images/screenshot_simulation_multiagent.jpg)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    pygame
    gymnasium
    numpy
    pyyaml
    casadi
    # Add any other specific RL libraries if/when implemented (e.g., stable-baselines3, torch)
    ```
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Navigate to the project's root directory in your terminal (where `main.py` is located) and run:

```bash
python main.py
```

This will launch the main menu, allowing you to select the desired simulation or training mode and configure agents.

## Configuration

The simulation behavior is primarily controlled through two YAML configuration files:

*   `config.yaml`: Used for multi-agent simulations and training.
*   `config_single_agent.yaml`: Used for single-agent simulations and training. It also contains the `agent_config` section detailing parameters for different agent types when run in single mode.

Key configuration sections:

*   **`simulation`**: General parameters like timestep (`dt`), number of episodes, max steps per episode, simulation `mode` ('single' or 'multi'), number of agents (`num_agents` - relevant for multi), `render_mode` ('human', 'rgb_array', or `None`), and `render_fps`.
*   **`training`**: Parameters specific to training runs (currently placeholders as RL training isn't fully implemented).
*   **`track`**: Defines the track geometry (e.g., `type`, `length`, `radius`, `width`). Currently supports 'oval'.
*   **`environment`**: Specifies which components of the car's state are included in the observation space (e.g., `x`, `y`, `v`, `theta`, `dist_to_centerline`).
*   **`car`**: Defines the physical parameters of the cars (e.g., `wheelbase`, `mass`, `max_speed`, force coefficients, dimensions).
*   **`agents`**: Defines the agents participating in the simulation.
    *   In `config.yaml` (multi-agent): Each key (e.g., `agent_0`, `agent_1`) defines an agent's `type` and starting position index (`start_pos_idx`).
    *   In `config_single_agent.yaml`: Contains only `agent_0` mainly for defining the `start_pos_idx`. The actual agent type is selected via the menu and parameters are merged from `agent_config`.
*   **`agent_config`** (in `config_single_agent.yaml`): Contains specific parameters for each agent type ('human', 'random', 'mpc', 'rl') used when running in single-agent mode.
*   **`rl`**: Hyperparameters for Reinforcement Learning (currently placeholders).

Modify these files to customize the simulation environment, agent behaviors, and track layout.

## Project Structure

```
.                     # Project Root
├── agents/           # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py   # Abstract base class for agents
│   ├── human_agent.py  # Keyboard controlled agent
│   ├── mpc_agent.py    # Model Predictive Control agent (using CasADi)
│   ├── random_agent.py # Agent taking random actions
│   └── rl_agent.py     # Placeholder for Reinforcement Learning agent
├── images/           # Contains images like the menu background
│   └── main_menu_background.png
├── models/           # Directory for saving/loading trained models (optional)
├── config.yaml       # Configuration for multi-agent mode
├── config_single_agent.yaml # Configuration for single-agent mode
├── car.py            # Car physics and state implementation
├── environment.py    # Gymnasium environment class (RacingEnv)
├── main.py           # Main entry point, runs the menu and starts simulation/training
├── menu.py           # Implements the Pygame main menu (MainMenu class)
├── README.md         # This file
├── requirements.txt  # Python dependencies (You need to create this)
├── simulation.py     # Contains the simulation loop logic (run_simulation, run_episode)
├── track.py          # Track geometry and related functions (e.g., collision, lap check)
├── training.py       # Placeholder functions for training RL agents
├── ui.py             # Pygame UI rendering for the simulation (UI class)
└── utils.py          # Utility functions (e.g., collision checking)

```

## Agent Types

*   **Human:** Controlled using the keyboard arrow keys (Up/Down for throttle/brake, Left/Right for steering).
*   **Random:** Selects actions randomly from the allowed action space.
*   **MPC (Model Predictive Control):** Uses CasADi to solve an optimization problem over a prediction horizon to determine the best action based on a kinematic bicycle model.
*   **RL (Reinforcement Learning):** Placeholder agent. Requires implementation of model loading (e.g., PyTorch, TensorFlow) and inference logic within `agents/rl_agent.py`.

## Notes

*   The Reinforcement Learning training and agent inference parts (`training.py`, parts of `agents/rl_agent.py`) are currently placeholders and need to be implemented using a specific RL library (like Stable Baselines3, Tianshou, or a custom implementation). The current `RLAgent` returns a default action.
*   The car physics model is a simplified kinematic bicycle model with added force dynamics. It may not perfectly reflect real-world vehicle behavior.

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
