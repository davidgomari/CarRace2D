# main.py
import yaml
import sys

from simulation import run_simulation
from training import single_agent_training, multi_agent_training
from menu import MainMenu

def load_config(mode):
    """Load the appropriate configuration file based on mode."""
    config_path = 'config_single_agent.yaml' if mode == 'single' else 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to handle both simulation and training modes."""
    while True:
        # Create and run the main menu
        menu = MainMenu()
        result = menu.run()
        menu.close()
        
        # If no result (user closed the window), exit
        if result is None:
            return
        
        # Process the menu result
        mode = result.get('mode')
        training = result.get('training', False)
        agent_type = result.get('agent_type')
        
        if training:
            # Training mode
            config = load_config(mode)
            if mode == 'single':
                single_agent_training(config)
            else:
                multi_agent_training(config)
        else:
            # Simulation mode
            if mode == 'single':
                # For single agent simulation, we need to set the agent type in the config
                config = load_config('single')
                # Update the agent type in the config
                if 'agents' in config and 'agent_0' in config['agents']:
                    config['agents']['agent_0']['type'] = agent_type
                # Run the simulation with the updated config
                run_simulation(config)
            else:
                # For multi agent simulation, use the default config
                config = load_config('multi')
                run_simulation(config)
        
        # After simulation or training completes, the loop will continue
        # and show the main menu again

if __name__ == "__main__":
    main()