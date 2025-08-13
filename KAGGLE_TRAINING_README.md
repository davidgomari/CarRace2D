# Kaggle Training Scripts for CarRace2D

This directory contains training scripts designed to run in Kaggle environments without requiring GUI interaction.

## Overview

Two main training scripts are provided:
1. **`train_single_agent.py`** - Train a single RL agent
2. **`train_multi_agent.py`** - Train multiple agents (including RL agents)

## Quick Start

### Single Agent Training

```bash
# Basic training with default configuration
python train_single_agent.py

# Training with Kaggle-optimized configuration
python train_single_agent.py --config config_kaggle_single.yaml

# Custom training parameters
python train_single_agent.py --episodes 1000 --learning-rate 0.002 --max-steps 3000
```

### Multi-Agent Training

```bash
# Basic training with default configuration
python train_multi_agent.py

# Training with Kaggle-optimized configuration
python train_multi_agent.py --config config_kaggle_multi.yaml

# Train all agents as RL agents
python train_multi_agent.py --agent-type rl

# Custom training parameters
python train_multi_agent.py --episodes 500 --learning-rate 0.001 --save-frequency 50
```

## Configuration Files

### Default Configurations
- `config_single_agent.yaml` - Default single agent configuration
- `config.yaml` - Default multi-agent configuration

### Kaggle-Optimized Configurations
- `config_kaggle_single.yaml` - Optimized for single agent training in Kaggle
- `config_kaggle_multi.yaml` - Optimized for multi-agent training in Kaggle

**Key optimizations for Kaggle:**
- `render_mode: None` - No GUI rendering
- Reduced episode counts and max steps for faster training
- Simplified physics parameters
- More frequent model saving
- Conservative learning rates

## Command Line Arguments

### Common Arguments (Both Scripts)
- `--config` - Path to configuration file
- `--episodes` - Override number of training episodes
- `--learning-rate` - Override learning rate
- `--max-steps` - Override max steps per episode
- `--save-frequency` - Override save frequency
- `--resume` - Resume training from existing models

### Multi-Agent Specific
- `--agent-type` - Override all agents to be of this type (rl, mpc, random)

## Example Usage in Kaggle

### Single Agent Training Example
```python
# In a Kaggle notebook cell
!python train_single_agent.py --config config_kaggle_single.yaml --episodes 200 --learning-rate 0.001
```

### Multi-Agent Training Example
```python
# In a Kaggle notebook cell
!python train_multi_agent.py --config config_kaggle_multi.yaml --agent-type rl --episodes 150
```

## Model Output

### Single Agent
- Models are saved to: `models/single_agent/trained_model.zip`

### Multi-Agent
- Models are saved to: `models/multi_agent/agent_X_model.zip`
- Each agent gets its own model file

## Training Progress

The scripts provide detailed progress information:
- Episode number and total reward
- Average reward over last 20 episodes
- Final progress on track
- Loss values for RL agents
- Model saving confirmations

## Resuming Training

To resume training from a previous checkpoint:

```bash
# Single agent
python train_single_agent.py --resume

# Multi-agent
python train_multi_agent.py --resume
```

**Note:** Make sure the model files exist in the expected locations before using `--resume`.

## Environment Requirements

The scripts require the same dependencies as the main project:
- PyTorch
- NumPy
- PyYAML
- Pygame (for environment, but not for GUI)
- Other dependencies listed in `requirements.txt`

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

### Performance Tips for Kaggle

1. **Use Kaggle-optimized configs** - They're designed for faster training
2. **Start with fewer episodes** - 100-500 episodes for initial testing
3. **Monitor GPU usage** - Use `nvidia-smi` if available
4. **Save frequently** - Use `--save-frequency 50` or lower
5. **Use conservative learning rates** - 0.001 or lower for stability

## Integration with Main Project

After training, you can use the trained models with the main simulation:

```python
# Load and use trained single agent model
from main import load_config
config = load_config('single')
config['agents']['agent_0']['type'] = 'rl'
# Run simulation with trained model
```

## Advanced Configuration

### Custom Agent Types
You can modify the configuration files to use different agent types:
- `rl` - Reinforcement Learning
- `mpc` - Model Predictive Control
- `random` - Random actions
- `human` - Human control (not suitable for Kaggle)

### Hyperparameter Tuning
Key parameters to tune:
- `learning_rate` - Start with 0.001, adjust based on convergence
- `discount_factor` - Usually 0.9-0.99
- `num_episodes` - More episodes = better learning but longer training
- `max_steps` - Longer episodes = more complex behaviors

### Observation Space
Modify `observation_components` in the config to change what the RL agent observes:
- `v` - Velocity
- `steer_angle` - Current steering angle
- `accel` - Current acceleration
- `dist_to_centerline` - Distance to track centerline
- `lidar` - LiDAR sensor readings (multi-agent only)

## Testing Trained Models

Use the `test_trained_model.py` script to validate your trained models:

```bash
# Test single agent model
python test_trained_model.py --mode single --episodes 5

# Test multi-agent models
python test_trained_model.py --mode multi --episodes 3
```

## File Structure

```
CarRace2D/
├── train_single_agent.py          # Single agent training script
├── train_multi_agent.py           # Multi-agent training script
├── test_trained_model.py          # Model testing script
├── config_kaggle_single.yaml      # Kaggle-optimized single agent config
├── config_kaggle_multi.yaml       # Kaggle-optimized multi-agent config
├── KAGGLE_TRAINING_README.md      # This file
├── models/
│   ├── single_agent/              # Single agent trained models
│   └── multi_agent/               # Multi-agent trained models
└── ... (other project files)
```

## Quick Reference

### Single Agent Training Commands
```bash
# Quick start with Kaggle config
python train_single_agent.py --config config_kaggle_single.yaml

# Custom training
python train_single_agent.py --episodes 300 --learning-rate 0.001 --max-steps 2000

# Resume training
python train_single_agent.py --resume
```

### Multi-Agent Training Commands
```bash
# Train all agents as RL
python train_multi_agent.py --agent-type rl --config config_kaggle_multi.yaml

# Custom multi-agent training
python train_multi_agent.py --episodes 200 --save-frequency 25

# Resume multi-agent training
python train_multi_agent.py --resume
```

### Testing Commands
```bash
# Test single agent
python test_trained_model.py --mode single

# Test multi-agent
python test_trained_model.py --mode multi

# Test with custom episodes
python test_trained_model.py --mode single --episodes 10
```
