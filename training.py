import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.save_util import save_to_zip_file
from agents.rl_agent import RLAgent
from simulation import setup_environment

def get_training_mode():
    """Get the training mode from user input."""
    mode = ''
    while mode not in ['single', 'multi']:
        mode = input("Train in 'single' or 'multi' agent mode? ").lower().strip()
    return mode

def save_model(model, mode, agent_id=None):
    """Save the trained model to the appropriate directory."""
    # Create models directory if it doesn't exist
    base_dir = 'models/single_agent' if mode == 'single' else 'models/multi_agent'
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate filename
    if mode == 'single':
        filename = os.path.join(base_dir, 'trained_model.zip')
    else:
        filename = os.path.join(base_dir, f'{agent_id}_model.zip')
    
    # Save the model
    # for savving, in future, use pickle
    # save_to_zip_file(filename, model)
    print(f"Model saved to {filename}")

def single_agent_training(config):
    """Train a single RL agent."""
    # Set up environment
    env, agents = setup_environment('single', 'rl', config)
    
    # Get the RL agent
    rl_agent = agents['agent_0']
    
    # Create and train the model
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=config['rl']['learning_rate'],
    #     gamma=config['rl']['discount_factor'],
    #     verbose=1
    # )
    model = None
    
    # Train the model
    num_episodes = config['training']['num_episodes']
    total_timesteps = num_episodes * config['simulation']['max_steps']
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    save_model(model, 'single')
    
    env.close()
    print("Single agent training completed!")

def multi_agent_training(config):
    """Train multiple RL agents in a multi-agent environment."""
    # Set up environment
    env, agents = setup_environment('multi', None, config)
    
    # Train each RL agent
    for agent_id, agent in agents.items():
        if isinstance(agent, RLAgent):
            print(f"\nTraining agent {agent_id}...")
            
            # Create and train the model
            # model = PPO(
            #     "MlpPolicy",
            #     env,
            #     learning_rate=config['rl']['learning_rate'],
            #     gamma=config['rl']['discount_factor'],
            #     verbose=1
            # )
            model = None
            
            # Train the model
            num_episodes = config['training']['num_episodes']
            total_timesteps = num_episodes * config['simulation']['max_steps']
            model.learn(total_timesteps=total_timesteps)
            
            # Save the model
            save_model(model, 'multi', agent_id)
    
    env.close()
    print("Multi-agent training completed!") 