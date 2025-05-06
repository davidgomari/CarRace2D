import os
import torch
from torch.optim import Adam
from agents.rl_agent import RLAgent
from simulation import setup_environment, handle_pygame_events, initialize_metrics
from rl_algo.reinforce import ReinforcePolicy
import yaml # Added for reloading config

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
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def single_agent_training(config):
    """Train a single RL agent."""
    # Set training mode flag
    config['training_mode'] = True
    
    # Set up environment
    # --- Use the training render mode specifically ---
    training_render_mode = config.get('training', {}).get('render_mode', None) # Get training render mode
    original_sim_render_mode = config.get('simulation', {}).get('render_mode') # Store original sim mode
    
    # --- Use the training max_steps specifically ---
    training_max_steps = config.get('training', {}).get('max_steps', 1000) # Default if missing
    original_sim_max_steps = config.get('simulation', {}).get('max_steps')
    
    # Temporarily override simulation render mode for environment setup
    if 'simulation' not in config: config['simulation'] = {}
    config['simulation']['render_mode'] = training_render_mode 
    config['simulation']['max_steps'] = training_max_steps
    
    print(f"Setting up training environment with render_mode: {training_render_mode}, max_steps: {training_max_steps}") 
    env, agents = setup_environment('single', 'rl', config)
    
    # --- Restore original simulation render mode in the config object ---
    config['simulation']['render_mode'] = original_sim_render_mode
    config['simulation']['max_steps'] = original_sim_max_steps
    # -------------------------------------------------

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Get the RL agent
    rl_agent = agents['agent_0']
    
    rl_algo = config['training']['rl_algo']
    # --- Initialize or Load Model (Algorithm Independent) --- 
    model = None # Use a local variable first
    model_path = config['agent_config']['rl']['model_path']
    resume_training = config['training']['resume_training']
    obs_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
    action_dim = env.action_space.shape[0]
    
    if resume_training:
        if os.path.exists(model_path):
            try:
                print(f"Attempting to resume training from {model_path}")
                # We need to know the model type to instantiate before loading state_dict
                # For now, assume ReinforcePolicy, but this needs generalization for other algos
                if rl_algo == 'reinforce':
                    model = ReinforcePolicy(obs_dim, action_dim)
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.to(device)
                    print(f"Successfully loaded model from {model_path} for resuming.")
                else:
                    # Placeholder: Add logic here to load models for other algorithms (PPO, SAC, etc.)
                    print(f"Warning: Resume logic not yet implemented for algorithm '{rl_algo}'. Starting fresh.")
                    pass # Fall through to create new model
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path} for resuming: {e}. Starting fresh.")
                model = None # Ensure we create a new one below
        else:
            print(f"Warning: Resume training enabled, but model file not found at {model_path}. Starting fresh.")
            
    # If model is still None (resume_training=False or loading failed/not found/not implemented)
    if model is None:
        print(f"Initializing a new {rl_algo} model for training.")
        if rl_algo == 'reinforce':
            model = ReinforcePolicy(obs_dim, action_dim).to(device)
        else:
            # Placeholder: Add logic here to create models for other algorithms
            raise ValueError(f"Model creation not implemented for algorithm: {rl_algo}")
            
    # Assign the created/loaded model to the agent
    rl_agent.model = model
    # ----------------------------------------------------------
    
    if rl_algo == 'reinforce':
        # Initialize Reinforce-specific optimizer
        optimizer = Adam(rl_agent.model.parameters(), lr=config['training']['learning_rate'])
        
    else:
        raise ValueError(f"Unsupported RL algorithm: {rl_algo}")
    
    # Train the model
    num_episodes = config['training']['num_episodes']
    max_steps = config['training']['max_steps']
    discount_factor = config['training']['discount_factor']

    # --- Tracking for Average Reward ---
    from collections import deque
    recent_rewards = deque(maxlen=20) # Store rewards of last 20 episodes
    # -----------------------------------

    # Run the training episodes with visualization
    print(f"Starting training with {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        ep_rewards = []
        ep_log_probs = []
        total_reward = 0

        step_idx = 0
        quit_signal_received = False # Flag to break outer loop
        
        # Initialize metrics for this episode
        metrics = initialize_metrics(agents)
        
        while not done:
            # --- Handle Pygame Events ---
            if env.render_mode == 'human':
                running, state, metrics, step_idx = handle_pygame_events(env, state, metrics, step_idx)
                if running == -1:  # Back button
                    print("Back button pressed. Saving model and stopping training.")
                    save_model(rl_agent.model, 'single')
                    env.close()
                    return
                elif running == -2: # Pause
                    continue # Skip the rest of the loop iteration (pause)
                elif not running: # Window closed
                    print("Window closed. Stopping training.")
                    env.close()
                    return
            # --- End Handle Pygame Events ---

            action, log_prob = rl_agent.model.act(state, episode)
            next_observation, reward, done, truncated, info = env.step(action)
            state = next_observation
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            total_reward += reward
            step_idx += 1
            
            # Check for quit signal from environment (user closed window)
            if info.get('quit_signal', False):
                print("Quit signal received, stopping training.")
                quit_signal_received = True
                break # Exit inner while loop
                
            # Check max steps (Gymnasium uses truncated now)
            if truncated:
                done = True # Use done flag to exit loop, truncated handles reason
            elif step_idx >= max_steps:
                done = True # Should be handled by truncated, but good failsafe

        # If user quit, break the outer episode loop as well
        if quit_signal_received:
            break
            
        # Compute returns (discounted cumulative rewards)
        returns = []
        R = 0
        for r in reversed(ep_rewards):
            R = r + discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        # Normalize returns for improved stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy loss (we want to maximize rewards)
        loss = 0
        for log_prob, R in zip(ep_log_probs, returns):
            loss -= log_prob * R  # gradient ascent on expected reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Update and Print Average Reward ---
        recent_rewards.append(total_reward)
        avg_reward_last_20 = sum(recent_rewards) / len(recent_rewards)
        # ---------------------------------------
        
        # Access the car's final progress
        final_progress = env.cars['agent_0'].total_progress
        print(f"Episode {episode+1:3d}: Total Reward = {total_reward:8.2f}, Avg Reward (Last 20) = {avg_reward_last_20:8.2f}, Final Progress = {final_progress:.4f}")
       
    
    # Save the model
    save_model(rl_agent.model, 'single')
    
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