#!/usr/bin/env python3
"""
Test Trained Model Script
This script loads and tests a trained model without GUI interaction.
Suitable for validating training results in Kaggle environments.
"""

import yaml
import argparse
import sys
from simulation import setup_environment, run_simulation

def load_config(config_path):
    """Load the configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def test_single_agent_model(model_path, config_path='config_single_agent.yaml', episodes=1):
    """Test a single agent trained model."""
    print("=" * 60)
    print("Testing Single Agent Trained Model")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up for testing (no training mode)
    config['training_mode'] = False
    config['simulation']['render_mode'] = None  # No GUI for testing
    config['simulation']['num_episodes'] = episodes
    
    # Set agent to RL and specify model path
    config['agents']['agent_0']['type'] = 'rl'
    config['agent_config']['rl']['model_path'] = model_path
    
    print(f"Testing model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Render mode: {config['simulation']['render_mode']}")
    
    try:
        # Set up environment and run simulation
        env, agents = setup_environment('single', 'rl', config)
        run_simulation(config)
        print("\nModel testing completed successfully!")
        
    except Exception as e:
        print(f"\nModel testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_multi_agent_models(model_paths, config_path='config.yaml', episodes=1):
    """Test multi-agent trained models."""
    print("=" * 60)
    print("Testing Multi-Agent Trained Models")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up for testing (no training mode)
    config['training_mode'] = False
    config['simulation']['render_mode'] = None  # No GUI for testing
    config['simulation']['num_episodes'] = episodes
    
    # Set all agents to RL and specify model paths
    for agent_id, model_path in model_paths.items():
        if agent_id in config['agents']:
            config['agents'][agent_id]['type'] = 'rl'
            config['agents'][agent_id]['model_path'] = model_path
            print(f"Agent {agent_id}: {model_path}")
    
    print(f"Episodes: {episodes}")
    print(f"Render mode: {config['simulation']['render_mode']}")
    
    try:
        # Set up environment and run simulation
        env, agents = setup_environment('multi', None, config)
        run_simulation(config)
        print("\nModel testing completed successfully!")
        
    except Exception as e:
        print(f"\nModel testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function for testing trained models."""
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--mode', choices=['single', 'multi'], required=True,
                       help='Test mode: single or multi agent')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to single agent model (for single mode)')
    parser.add_argument('--model-paths', type=str, nargs='+', default=None,
                       help='Paths to multi-agent models (for multi mode)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to test')
    
    args = parser.parse_args()
    
    # Set default config paths
    if args.config is None:
        args.config = 'config_single_agent.yaml' if args.mode == 'single' else 'config.yaml'
    
    if args.mode == 'single':
        # Single agent testing
        if args.model_path is None:
            args.model_path = 'models/single_agent/trained_model.zip'
        test_single_agent_model(args.model_path, args.config, args.episodes)
    
    else:
        # Multi-agent testing
        if args.model_paths is None:
            # Use default model paths
            model_paths = {
                'agent_0': 'models/multi_agent/agent_0_model.zip',
                'agent_1': 'models/multi_agent/agent_1_model.zip'
            }
        else:
            # Parse model paths from command line
            model_paths = {}
            for i, path in enumerate(args.model_paths):
                model_paths[f'agent_{i}'] = path
        
        test_multi_agent_models(model_paths, args.config, args.episodes)

if __name__ == "__main__":
    main()
