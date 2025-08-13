# CarRace2D Simulation

A comprehensive 2D racing simulation environment built with Python, Pygame, and Gymnasium. Perfect for reinforcement learning research, autonomous vehicle development, and multi-agent racing simulations. **Version 2.5** introduces headless training capabilities for cloud environments, enhanced LiDAR sensor support, and improved MPC agent performance.

## 🚀 Quick Start

### Installation

1. **Clone and setup:**
   ```bash
   git clone <your-repository-url>
   cd CarRace2D
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python main.py
   ```

3. **Choose your mode:**
   - **Simulation**: Watch agents race with real-time visualization
   - **Training**: Train RL agents with interactive monitoring

### First Steps

1. **Try Human Control:**
   - Select "Simulation" → "Single Agent" → "Human"
   - Use arrow keys: ↑↓ (throttle/brake), ←→ (steering)

2. **Watch AI Agents:**
   - Select "Simulation" → "Single Agent" → "MPC" or "Random"
   - Observe how different agent types behave

3. **Train Your First RL Agent:**
   - Select "Training" → "Single Agent"
   - Monitor training progress in real-time
   - Use UI controls to pause/resume/save

## 🎯 Key Features

### 🏎️ **Multiple Agent Types**
- **Human**: Keyboard-controlled racing
- **RL**: Reinforcement Learning agents (REINFORCE algorithm)
- **MPC**: Model Predictive Control with collision avoidance
- **Random**: Baseline random behavior

### 🌐 **Flexible Environments**
- **Single Agent**: Focus on individual agent performance
- **Multi-Agent**: Competitive racing with multiple agents
- **Headless Training**: Cloud-ready training without GUI

### 🔧 **Advanced Systems**
- **LiDAR Sensor**: Configurable sensor system for multi-agent scenarios
- **Modular MPC**: Enhanced control with collision avoidance
- **Real-time UI**: Interactive controls and monitoring
- **Configurable Physics**: Realistic vehicle dynamics

## 📁 Project Structure

```
CarRace2D/
├── main.py                    # Main entry point
├── train_single_agent.py      # Headless single agent training
├── train_multi_agent.py       # Headless multi-agent training
├── test_trained_model.py      # Model validation script
├── config_single_agent.yaml   # Single agent configuration
├── config.yaml               # Multi-agent configuration
├── config_kaggle_*.yaml      # Cloud-optimized configs
├── agents/                   # Agent implementations
├── models/                   # Trained model storage
├── images/                   # UI assets
└── KAGGLE_TRAINING_README.md # Cloud training guide
```

## 🎮 Usage Modes

### GUI Mode (Local Development)
```bash
python main.py
```
- Interactive menu system
- Real-time visualization
- Training monitoring
- Model management

### Headless Mode (Cloud/Kaggle)
```bash
# Single agent training
python train_single_agent.py --config config_kaggle_single.yaml --episodes 200

# Multi-agent training
python train_multi_agent.py --config config_kaggle_multi.yaml --agent-type rl --episodes 150

# Test trained models
python test_trained_model.py --mode single --episodes 5
```

**📖 See [KAGGLE_TRAINING_README.md](KAGGLE_TRAINING_README.md) for detailed cloud training instructions.**

## ⚙️ Configuration

The simulation is highly configurable through YAML files:

- **`config_single_agent.yaml`**: Single agent settings
- **`config.yaml`**: Multi-agent settings  
- **`config_kaggle_*.yaml`**: Cloud-optimized configurations

Key configuration areas:
- **Simulation**: Episode count, time steps, rendering
- **Training**: Learning rates, algorithms, save frequency
- **Physics**: Vehicle parameters, track layout
- **Agents**: Behavior types, model paths
- **UI**: Display options, sidebar customization

## 🎯 Common Use Cases

### Research & Development
- **RL Algorithm Testing**: Implement and test new algorithms
- **Multi-Agent Learning**: Study competitive behaviors
- **Control Systems**: Develop and validate MPC strategies
- **Sensor Fusion**: Experiment with LiDAR and other sensors

### Education & Learning
- **Reinforcement Learning**: Learn RL concepts hands-on
- **Control Theory**: Understand MPC and optimal control
- **Game AI**: Study agent behavior and decision making
- **Physics Simulation**: Explore vehicle dynamics

### Competition & Benchmarking
- **Agent Comparison**: Compare different AI approaches
- **Performance Testing**: Benchmark algorithms and strategies
- **Racing Competitions**: Host multi-agent racing events

## 🛠️ Customization

### Adding New Agents
1. Create agent class in `agents/` directory
2. Inherit from `BaseAgent`
3. Implement `get_action()` method
4. Register in configuration

### Modifying Physics
- Edit `car.py` for vehicle dynamics
- Adjust parameters in config files
- Add new physics components

### Creating Custom Tracks
- Implement track class in `track.py`
- Add visualization in `ui.py`
- Configure in YAML files

### Implementing New RL Algorithms
- Add algorithm file in `rl_algo/` directory
- Follow existing interface
- Register in training functions

## 📊 Performance Tips

### For Training
- Use `render_mode: None` for faster training
- Start with fewer episodes (100-500) for testing
- Use Kaggle-optimized configs for cloud environments
- Monitor GPU usage with `nvidia-smi`

### For Simulation
- Reduce `render_fps` for better performance
- Use simplified physics for large agent counts
- Optimize track complexity based on needs

### For Development
- Use debug scripts for troubleshooting
- Test with small configurations first
- Save models frequently during development

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Black screen/No display | Update graphics drivers, use headless mode for training |
| Slow performance | Reduce render_fps, use `render_mode: None` |
| Import errors | Check virtual environment, install requirements |
| Memory issues | Reduce agent count, simplify physics |
| LiDAR problems | Use `test_lidar_debug.py` script |

### Getting Help
1. Check the [Documentation](DOCUMENTATION.md) for detailed guides
2. Review configuration examples
3. Use debug scripts for troubleshooting
4. Check error messages for specific issues

## 📚 Documentation

<div align="center">
  <h3>📖 Comprehensive Documentation Available</h3>
  <p>
    <a href="DOCUMENTATION.md">
      <img src="https://img.shields.io/badge/Documentation-Click%20Here-blue?style=for-the-badge&logo=markdown&logoColor=white" alt="Documentation Badge">
    </a>
  </p>
</div>

**📋 Documentation includes:**
- Detailed configuration guides
- Customization tutorials
- API reference
- Best practices
- Advanced features
- Performance optimization
- Troubleshooting guides

**[📖 View Full Documentation](DOCUMENTATION.md)**

## 🚀 What's New in v2.5

### ✨ Major Features
- **🌐 Headless Training**: Cloud-ready training scripts for Kaggle and other environments
- **📡 Enhanced LiDAR**: Configurable sensor system with collision detection
- **🎯 Improved MPC**: Modular cost functions and collision avoidance
- **⚡ Performance**: Optimized configurations for faster training
- **🐛 Debug Tools**: Enhanced error handling and debugging scripts

### 🔧 Improvements
- Better error messages and validation
- More flexible configuration options
- Enhanced documentation and examples
- Improved training stability
- Better cloud environment support

## 🗺️ Roadmap

### ✅ Completed
- [x] Multi-agent RL training support
- [x] Headless training for cloud environments
- [x] Enhanced LiDAR sensor system
- [x] Improved MPC agent with modular components
- [x] Real-time UI with interactive controls

### 🚧 Planned
- [ ] Additional RL algorithms (PPO, TD3, SAC)
- [ ] Custom track editor
- [ ] Replay system for saving/loading races
- [ ] Performance optimization for large-scale simulations
- [ ] Additional physics models and vehicle types

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pygame](https://www.pygame.org/) and [Gymnasium](https://gymnasium.farama.org/)
- Enhanced with [CasADi](https://web.casadi.org/) for MPC optimization
- Powered by [PyTorch](https://pytorch.org/) for reinforcement learning
- Inspired by autonomous racing research and development

---

**🎯 Ready to start racing? Run `python main.py` and choose your adventure!**
