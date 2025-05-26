import numpy as np
import yaml
from environment import RacingEnv
import sys

available_agents_types = ['human', 'rl', 'mpc', 'random']

def load_config(mode):
    """Load the appropriate configuration file based on mode."""
    config_path = 'config_single_agent.yaml' if mode == 'single' else 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_simulation_mode():
    """Get the simulation mode from user input."""
    mode = ''
    while mode not in ['single', 'multi']:
        mode = input("Run in 'single' or 'multi' agent mode? ").lower().strip()
    return mode

def get_agent_type():
    """Get the agent type from user input for single-agent mode."""
    agent_type = None
    while agent_type not in available_agents_types:
        agent_type = str(input(f"Enter the agent type ({available_agents_types}): "))
        if agent_type not in available_agents_types:
            print(f"Invalid agent type. Please choose from {available_agents_types}.")
    return agent_type

def setup_environment(mode, agent_type, config):
    """Set up the environment and agents based on configuration."""
    
    # Determine if we're in training mode
    is_training = config.get('training_mode', False)  # Default to simulation mode

    # --- Unify MPC agent config for all agents (single or multi) ---
    for agent_id, agent_cfg in config.get('agents', {}).items():
        a_type = agent_cfg.get('type')
        if a_type == 'mpc':
            # Use agent_config['mpc'] as base, then overlay agent-specific overrides
            merged = dict(config.get('agent_config', {}).get('mpc', {}))
            merged.update(agent_cfg)  # agent-specific overrides take precedence
            config['agents'][agent_id] = merged
        elif a_type == 'rl':
            agent_cfg['training_mode'] = is_training

    if mode == 'single':
        print(f"Running single-agent {'training' if is_training else 'simulation'}, requested type: {agent_type}")
        # Update the config for the single agent based on the chosen type
        if agent_type in config.get('agent_config', {}):
            # Set the agent type in the main 'agents' section
            config['agents']['agent_0']['type'] = agent_type
            # For non-MPC types, merge agent_config[agent_type] (MPC already handled above)
            if agent_type != 'mpc':
                agent_specific_config = config['agent_config'][agent_type]
                config['agents']['agent_0'].update(agent_specific_config)
            # Set training_mode based on whether we're in training
            if agent_type == 'rl':
                config['agents']['agent_0']['training_mode'] = is_training
            print(f"Loaded configuration for {agent_type} agent.")
        else:
            print(f"Warning: Configuration for agent type '{agent_type}' not found in agent_config.")
            print("Proceeding with default agent_0 config (if any) or environment defaults.")
            # Ensure 'type' is set even if config is missing, env might handle it
            config['agents']['agent_0']['type'] = agent_type 

    # Determine if any agent requires human render mode
    is_human_controlled = False
    active_agents_config = config.get('agents', {})
    if any(agent_info.get('type') == 'human' for agent_id, agent_info in active_agents_config.items()):
        print("Human agent detected, setting render_mode to 'human'.")
        config['simulation']['render_mode'] = 'human'
        is_human_controlled = True

    # --- Create the environment AFTER potentially modifying the config --- 
    try:
        env = RacingEnv(config=config, mode=mode) 
    except Exception as e:
        print(f"\nError initializing environment: {e}")
        print("Please check your configuration file and agent implementations.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    agents = env.agents

    # Print final confirmation
    if mode == 'single':
        print(f"Successfully set up single-agent env with type: {agents['agent_0'].__class__.__name__}")
    else:
        print(f"Running multi-agent simulation with {env.num_agents} agents:")
        for agent_id, agent in agents.items():
            print(f"  {agent_id}: {agent.__class__.__name__}")

    return env, agents

def initialize_metrics(agents):
    """Initialize performance metrics storage."""
    return {
        'episode_lap_times': {agent_id: [] for agent_id in agents.keys()},
        'episode_reward_sum': {agent_id: 0 for agent_id in agents.keys()},
        'current_lap_times': {agent_id: [] for agent_id in agents.keys()},
        'lap_start_times': {agent_id: 0.0 for agent_id in agents.keys()}
    }

def print_controls():
    """Print control instructions for human players."""
    print("\nControls:")
    print("Up Arrow: Accelerate")
    print("Down Arrow: Brake")
    print("Left Arrow: Turn Left")
    print("Right Arrow: Turn Right")
    print("ESC: Quit\n")

def get_actions(mode, agents, observations):
    """Get actions from agents based on mode."""
    if mode == 'single':
        return agents['agent_0'].get_action(observations)
    else:
        return {agent_id: agent.get_action(observations[agent_id]) 
                for agent_id, agent in agents.items()}

def update_metrics(mode, rewards, infos, current_time, metrics):
    """Update performance metrics based on rewards and lap completion."""
    if mode == 'single':
        metrics['episode_reward_sum']['agent_0'] += rewards
        agent_info = infos
        
        if agent_info.get('lap_completed', False):
            lap_time = current_time - metrics['lap_start_times']['agent_0']
            metrics['current_lap_times']['agent_0'].append(lap_time)
            metrics['lap_start_times']['agent_0'] = current_time
            print(f"Agent agent_0 Lap {agent_info.get('lap', '?')}: {lap_time:.2f}s")
    
    elif mode == 'multi':
        for agent_id in metrics['episode_reward_sum'].keys():
            metrics['episode_reward_sum'][agent_id] += rewards[agent_id]
            agent_info = infos.get(agent_id, {})
            
            if agent_info.get('lap_completed', False):
                lap_time = current_time - metrics['lap_start_times'][agent_id]
                metrics['current_lap_times'][agent_id].append(lap_time)
                metrics['lap_start_times'][agent_id] = current_time
                print(f"Agent {agent_id} Lap {agent_info.get('lap', '?')}: {lap_time:.2f}s")

def check_termination(mode, terminated, truncated):
    """Check if the episode should terminate."""
    if mode == 'single':
        return terminated or truncated
    return all(terminated.values()) or all(truncated.values())

def print_episode_summary(episode, current_step, metrics, agents):
    """Print summary of the completed episode."""
    print(f"--- Episode {episode + 1} Finished ({current_step} steps) ---")
    for agent_id in agents.keys():
        print(f"Agent {agent_id} Total Reward: {metrics['episode_reward_sum'][agent_id]:.2f}")
        if metrics['current_lap_times'][agent_id]:
            avg_lap_time = np.mean(metrics['current_lap_times'][agent_id])
            print(f"  Avg Lap Time: {avg_lap_time:.2f}s")
            metrics['episode_lap_times'][agent_id].extend(metrics['current_lap_times'][agent_id])
        else:
            print("  No laps completed.")

def print_final_analysis(metrics, agents):
    """Print final performance analysis."""
    print("\n--- Simulation Finished ---")
    print("Overall Average Lap Times:")
    for agent_id in agents.keys():
        if metrics['episode_lap_times'][agent_id]:
            overall_avg = np.mean(metrics['episode_lap_times'][agent_id])
            overall_std = np.std(metrics['episode_lap_times'][agent_id])
            print(f"Agent {agent_id}: Avg={overall_avg:.2f}s, StdDev={overall_std:.2f}s (from {len(metrics['episode_lap_times'][agent_id])} laps)")
        else:
            print(f"Agent {agent_id}: No laps completed across all episodes.")

def handle_pygame_events(env, observations, metrics, current_step):
    """Handle pygame events and return updated state variables."""
    running = env.ui.handle_events()
    
    # Check if we should return to the main menu
    if not running and env.ui.should_return_to_menu():
        return -1, observations, metrics, current_step  # Special value to indicate return to menu
    
    # Check if we should reset the environment
    if env.ui.should_reset:  # Access as a property, not a method
        observations, infos = env.reset()
        # Reset metrics for this episode
        for agent_id in metrics['episode_reward_sum'].keys():
            metrics['episode_reward_sum'][agent_id] = 0
            metrics['lap_start_times'][agent_id] = 0.0
            metrics['current_lap_times'][agent_id] = []
        current_step = 0
        # Reset the flag after handling it
        env.ui.should_reset = False
        return running, observations, metrics, current_step
    
    # Check if we're paused
    if env.ui.is_paused():
        # Just render the current state without progressing
        env.render()
        return -2, observations, metrics, current_step
    
    return running, observations, metrics, current_step

def run_episode(env, mode, agents, metrics, config):
    """Run a single episode of the simulation."""
    # Reset environment
    observations, infos = env.reset()
    
    # Reset episode-specific metrics
    for agent_id in agents.keys():
        metrics['episode_reward_sum'][agent_id] = 0
        metrics['lap_start_times'][agent_id] = 0.0
        metrics['current_lap_times'][agent_id] = []
    current_step = 0

    # Initialize pygame clock for controlling frame rate
    if env.render_mode == 'human':
        print_controls()

    # Main Simulation Loop
    running = True
    while running:
        # Handle Pygame Events
        if env.render_mode == 'human':
            running, observations, metrics, current_step = handle_pygame_events(
                env, observations, metrics, current_step)
            if running == -1:  # Special value to indicate return to menu
                return -1
            elif running == -2: # Special value to indicate pause
                continue
        
        if not running: break

        # Get Actions and Step Environment
        actions = get_actions(mode, agents, observations)
        observations, rewards, terminated, truncated, infos = env.step(actions)

        # Update Metrics
        current_step += 1
        current_time = current_step * env.dt
        update_metrics(mode, rewards, infos, current_time, metrics)

        # Render Frame
        if env.render_mode in ['human', 'rgb_array']:
            env.render()

        # Check Termination/Truncation
        if check_termination(mode, terminated, truncated):
            running = False

    return current_step

def run_simulation(config=None):
    """Main simulation function."""
    # Get simulation parameters if config not provided
    if config is None:
        mode = get_simulation_mode()
        config = load_config(mode)
        agent_type = get_agent_type() if mode == 'single' else None
    else:
        # Extract mode and agent_type from config
        mode = 'single' if config.get('simulation', {}).get('mode') == 'single' else 'multi'
        agent_type = None
        if mode == 'single' and 'agents' in config and 'agent_0' in config['agents']:
            agent_type = config['agents']['agent_0'].get('type')
    
    # Set up environment and agents
    env, agents = setup_environment(mode, agent_type, config)
    
    # Initialize metrics
    metrics = initialize_metrics(agents)
    
    # Run episodes
    num_episodes = config['simulation']['num_episodes']
    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        try:
            current_step = run_episode(env, mode, agents, metrics, config)
            
            # Check if we should return to the main menu
            if current_step == -1:
                print("Returning to main menu...")
                env.close()
                return
            
            print_episode_summary(episode, current_step, metrics, agents)
            
        except Exception as e:
            print(f"An error occurred during episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            sys.exit()

    env.close()
    print_final_analysis(metrics, agents)
    
    # Return to main menu after simulation is complete
    print("\nSimulation complete. Returning to main menu...")
    return 