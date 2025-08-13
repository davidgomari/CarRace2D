#!/usr/bin/env python3
"""
Multi-Agent Training Script
This script trains multiple agents (including RL agents) without requiring GUI interaction.
Suitable for running in Kaggle or other headless environments.
"""

import yaml
import argparse
import os
import sys
from training import multi_agent_training

def load_config(config_path='config.yaml'):
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

def validate_config(config):
    """Validate the configuration for multi-agent training."""
    required_sections = ['training', 'agents']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section '{section}'")
            sys.exit(1)
    
    # Ensure training render_mode is set to None for headless training
    if 'training' in config and 'render_mode' in config['training']:
        config['training']['render_mode'] = None
        print("Setting training render_mode to None for headless training")
    
    # Check if there are any agents configured
    if not config['agents']:
        print("Error: No agents configured in the agents section")
        sys.exit(1)
    
    # Count RL agents
    rl_agents = [agent_id for agent_id, agent_config in config['agents'].items() 
                 if agent_config.get('type') == 'rl']
    
    if not rl_agents:
        print("Warning: No RL agents configured. Training will only run non-RL agents.")
    else:
        print(f"Found {len(rl_agents)} RL agent(s): {rl_agents}")
    
    # Ensure all RL agents have model_path configured
    for agent_id, agent_config in config['agents'].items():
        if agent_config.get('type') == 'rl':
            if 'model_path' not in agent_config:
                # Set default model path
                agent_config['model_path'] = f'models/multi_agent/{agent_id}_model.zip'
                print(f"Setting default model path for {agent_id}: {agent_config['model_path']}")

def print_agent_configuration(config):
    """Print the agent configuration."""
    print("\nAgent Configuration:")
    for agent_id, agent_config in config['agents'].items():
        agent_type = agent_config.get('type', 'unknown')
        start_pos = agent_config.get('start_pos_idx', 0)
        model_path = agent_config.get('model_path', 'N/A')
        print(f"  {agent_id}:")
        print(f"    Type: {agent_type}")
        print(f"    Start Position: {start_pos}")
        if agent_type == 'rl':
            print(f"    Model Path: {model_path}")

def main():
    """Main function for multi-agent training."""
    parser = argparse.ArgumentParser(description='Train multiple agents')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate for RL agents')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Override max steps per episode')
    parser.add_argument('--save-frequency', type=int, default=None,
                       help='Override save frequency')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing models')
    parser.add_argument('--agent-type', type=str, choices=['rl', 'mpc', 'random'],
                       help='Override all agents to be of this type')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Agent Training")
    print("=" * 60)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate and prepare configuration
    validate_config(config)
    
    # Override agent types if specified
    if args.agent_type:
        for agent_id in config['agents']:
            config['agents'][agent_id]['type'] = args.agent_type
            if args.agent_type == 'rl':
                config['agents'][agent_id]['model_path'] = f'models/multi_agent/{agent_id}_model.zip'
        print(f"Overriding all agents to type: {args.agent_type}")
    
    # Apply command line overrides
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
        print(f"Overriding episodes to: {args.episodes}")
    
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
        print(f"Overriding learning rate to: {args.learning_rate}")
    
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
        print(f"Overriding max steps to: {args.max_steps}")
    
    if args.save_frequency is not None:
        config['training']['save_frequency'] = args.save_frequency
        print(f"Overriding save frequency to: {args.save_frequency}")
    
    if args.resume:
        config['training']['resume_training'] = True
        print("Enabling resume training")
    
    # Print agent configuration
    print_agent_configuration(config)
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Episodes: {config['training']['num_episodes']}")
    print(f"  Max Steps: {config['training']['max_steps']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Discount Factor: {config['training']['discount_factor']}")
    print(f"  RL Algorithm: {config['training']['rl_algo']}")
    print(f"  Save Frequency: {config['training'].get('save_frequency', 'N/A')}")
    print(f"  Resume Training: {config['training']['resume_training']}")
    print(f"  Render Mode: {config['training']['render_mode']}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/multi_agent', exist_ok=True)
    
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        # Start training
        multi_agent_training(config)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
