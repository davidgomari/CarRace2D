#!/usr/bin/env python3
"""
Single Agent RL Training Script
This script trains a single RL agent without requiring GUI interaction.
Suitable for running in Kaggle or other headless environments.
"""

import yaml
import argparse
import os
import sys
from training import single_agent_training

def load_config(config_path='config_single_agent.yaml'):
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
    """Validate the configuration for single agent training."""
    required_sections = ['training', 'agents', 'agent_config']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section '{section}'")
            sys.exit(1)
    
    # Ensure training render_mode is set to None for headless training
    if 'training' in config and 'render_mode' in config['training']:
        config['training']['render_mode'] = None
        print("Setting training render_mode to None for headless training")
    
    # Ensure agent_0 exists and is set to RL type
    if 'agent_0' not in config['agents']:
        print("Error: agent_0 not found in agents configuration")
        sys.exit(1)
    
    # Set agent type to RL for training
    config['agents']['agent_0']['type'] = 'rl'
    print("Setting agent_0 type to 'rl' for training")

def main():
    """Main function for single agent training."""
    parser = argparse.ArgumentParser(description='Train a single RL agent')
    parser.add_argument('--config', type=str, default='config_single_agent.yaml',
                       help='Path to configuration file (default: config_single_agent.yaml)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Override max steps per episode')
    parser.add_argument('--save-frequency', type=int, default=None,
                       help='Override save frequency')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Single Agent RL Training")
    print("=" * 60)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate and prepare configuration
    validate_config(config)
    
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
    os.makedirs('models/single_agent', exist_ok=True)
    
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        # Start training
        single_agent_training(config)
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
