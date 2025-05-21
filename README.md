# CarRace2D Simulation

A 2D racing simulation environment built with Python, Pygame, and Gymnasium. It supports single and multi-agent simulations with various agent types (Human, RL, MPC, Random) and features a graphical user interface.

## Features

*   **Graphical User Interface:** Main menu for selecting modes (Simulation/Training) and agent configurations using Pygame.
*   **Simulation Modes:** Supports both single-agent and multi-agent simulations.
*   **Agent Types:** Includes Human (keyboard control), Random, basic Model Predictive Control (MPC using CasADi), and Reinforcement Learning (RL) agents with configurable algorithms.
*   **Configurable Environment:** Simulation parameters, track layout, car physics, agent types, and more can be configured via YAML files.
*   **Modular Structure:** Code is organized into modules for UI, environment, track, car, agents, etc.
*   **Basic Physics:** Implements a kinematic bicycle model with considerations for engine/brake forces, drag, and rolling resistance.
*   **Training Support:** Built-in training functionality for RL agents with configurable parameters and visualization options.
*   **Multi-Agent RL Training:** Supports simultaneous training of multiple RL agents in a shared environment, with separate models and optimizers for each agent.
*   **UI Controls:** Interactive controls during simulation and training (Back, Reset, Pause/Resume).
*   **Customizable UI Sidebar:** Configure which agent/car properties are displayed in the sidebar via the config file.
*   **Flexible Observations:** Easily add new observation components (e.g., acceleration) to the environment and UI.

## Visuals

![Simulation - Multi Agent Mode](/images/screenshot_simulation_multiagent_v2.png)

## Requirements

* Python 3.8 or higher
* Operating System: Windows, macOS, or Linux
* Graphics: OpenGL 2.0 compatible graphics card
* Memory: 4GB RAM minimum, 8GB recommended
* Storage: 500MB free space

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
    torch
    # Add any other specific RL libraries if/when implemented (e.g., stable-baselines3)
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

## Quick Start Guide

### UI Controls

During both simulation and training, the following buttons are available:

*   **Back Button:** Returns to the main menu. During training, this will save the current model before exiting.
*   **Reset Button:** Resets the current episode, placing all agents back at their starting positions.
*   **Pause/Resume Button:** Toggles between pausing and resuming the simulation/training.
*   **Sidebar Columns:** You can customize which agent/car properties are shown in the sidebar by editing the `ui.sidebar_columns` section in your config file.

### Agent Types

*   **Human:** Controlled using keyboard arrow keys (Up/Down for throttle/brake, Left/Right for steering).
*   **Random:** Selects actions randomly from the allowed action space.
*   **MPC:** Uses CasADi to solve an optimization problem for determining the best action.
*   **RL:** Implements a configurable RL agent that can load trained models and use different RL algorithms.

### Training

To resume training from a previous model, set `resume_training: True` in your config file. Models are saved automatically during training.

**Multi-Agent Training:**

You can train multiple RL agents simultaneously in multi-agent mode. Each RL agent will have its own model and optimizer, and training progress is tracked individually. Configure multi-agent training in `config.yaml` and select "Multi Agent Training" from the main menu.

## Documentation

<div align="center">
  <h2>📚 Detailed Documentation Available</h2>
  <p>
    <a href="DOCUMENTATION.md">
      <img src="https://img.shields.io/badge/Documentation-Click%20Here-blue?style=for-the-badge&logo=markdown&logoColor=white" alt="Documentation Badge">
    </a>
  </p>
  <p>
    For comprehensive information about configuration, customization, and advanced features, please refer to our detailed documentation.
  </p>
</div>

The documentation includes:
- Detailed configuration guides
- Customization options for agents, physics, and tracks
- Advanced features and optimization tips
- Complete API reference
- Best practices and examples

[Click here to view the full documentation](DOCUMENTATION.md)

## Troubleshooting

Common issues and their solutions:

* **Black Screen/No Display**: Ensure your graphics drivers are up to date and OpenGL 2.0 is supported.
* **Slow Performance**: Try reducing the `render_fps` in the configuration file or setting `render_mode` to `None` for training.
* **Import Errors**: Make sure all dependencies are installed correctly in your virtual environment.
* **Memory Issues**: Reduce the number of agents or simulation complexity in the configuration file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

Planned features and improvements:

* [OK] Multi-agent RL training support
* [ ] Additional RL algorithms (PPO, TD3, SAC)
* [ ] Custom track editor
* [ ] Replay system for saving and loading races
* [ ] Performance optimization for large-scale simulations
* [ ] Additional physics models and vehicle types

## Acknowledgments

* Built with [Pygame](https://www.pygame.org/) and [Gymnasium](https://gymnasium.farama.org/)
