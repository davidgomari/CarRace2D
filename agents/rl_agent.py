# agents/rl_agent.py
# Base agent and numpy import
from .base_agent import BaseAgent
import numpy as np
import os # Import os for path checking
import torch # Import PyTorch
from rl_algo.reinforce import ReinforcePolicy # Import the policy network

class RLAgent(BaseAgent):
    def __init__(self, agent_id, agent_info):
        super().__init__(agent_id)
        self.model = None
        self.model_path = agent_info.get('model_path')
        self.device = agent_info.get('device', 'cpu') # Example: Allow config to specify device
        self.obs_dim = None
        self.action_dim = None

        if not self.model_path:
            raise ValueError(f"RL Agent {self.id}: 'model_path' must be specified in agent_info.")

        print(f"RL Agent {self.id}: Loading model from {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RL Agent {self.id}: Model file not found at {self.model_path}.")

        try:
            # Load state dictionary first to inspect it
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Infer dimensions from state_dict (assuming simple MLP structure)
            param_names = list(state_dict.keys())
            first_param = state_dict[param_names[0]]
            last_param = state_dict[param_names[-1]]
            
            # Infer obs_dim from the input features of the first layer's weights
            # Check if the first param is weights (2D) or bias (1D)
            if len(first_param.shape) == 2:
                self.obs_dim = first_param.shape[1]
            elif len(param_names) > 1 and len(state_dict[param_names[1]].shape) == 2:
                 # If first is bias, try second param (assuming it's the first weight matrix)
                 self.obs_dim = state_dict[param_names[1]].shape[1]
            else:
                raise ValueError("Could not infer obs_dim from model state_dict (first layer weights not found).")
                     
            # Infer action_dim from the output features of the last layer (bias or weights)
            self.action_dim = last_param.shape[0]
            
            print(f"RL Agent {self.id}: Inferred obs_dim={self.obs_dim}, action_dim={self.action_dim}")

            # Initialize the network structure with inferred dimensions
            self.model = ReinforcePolicy(self.obs_dim, self.action_dim) 
            
            # Load the state dictionary into the initialized model
            self.model.load_state_dict(state_dict)
            
            # Move model to the correct device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval() 
            print(f"RL Agent {self.id}: Successfully loaded model from {self.model_path} to device {self.device}")

        except Exception as e:
            print(f"RL Agent {self.id}: Failed to load model or infer dimensions from {self.model_path}. Error: {e}")
            self.model = None # Ensure model is None if loading fails
            # Re-raise the exception to prevent agent use without a model
            raise RuntimeError(f"RL Agent {self.id} initialization failed.") from e

    def _prepare_observation_vector(self, observation):
        """Converts observation (dict or array) into a flat numpy array."""
        if isinstance(observation, dict):
            # Use dict.keys() order - assuming this matches training
            try:
                obs_values = [observation[key] for key in observation.keys()]
                obs_vector = np.concatenate([np.atleast_1d(v).astype(np.float32) for v in obs_values])
            except Exception as e:
                 print(f"RL Agent {self.id} Error: Failed to convert observation dict to vector: {e}. Obs: {observation}")
                 raise ValueError("Observation dictionary conversion failed.") from e
        elif isinstance(observation, np.ndarray):
            obs_vector = observation.astype(np.float32)
        else: # Handle lists or other types
            try:
                obs_vector = np.array(observation, dtype=np.float32)
            except Exception as e:
                 print(f"RL Agent {self.id} Error: Failed to convert observation to numpy array: {e}. Obs: {observation}")
                 raise ValueError("Observation conversion to numpy array failed.") from e
        
        # Ensure the vector is flat and has the correct dimension
        obs_vector = obs_vector.flatten()
        if obs_vector.size != self.obs_dim:
            raise ValueError(f"RL Agent {self.id} Error: Observation dimension mismatch. Expected {self.obs_dim}, Got {obs_vector.size}. Original Obs: {observation}")
            
        return obs_vector

    def get_action(self, observation):
        """ Gets action based on the observation dictionary. """
        if not self.model:
            # This should not happen if __init__ succeeded, but added as a safeguard
            print(f"RL Agent {self.id}: Error - get_action called but model is not loaded. Returning default action.")
            fallback_action_dim = self.action_dim if self.action_dim is not None else 2
            return np.array([0.0] * fallback_action_dim, dtype=np.float32)
        
        try:
            # 1. Prepare observation vector
            obs_vector = self._prepare_observation_vector(observation)
            obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 2. Get action from model (deterministic mean)
            with torch.no_grad():
                # Assume model forward returns tuple (mean, std)
                action_mean, _ = self.model(obs_tensor)
                action = action_mean.squeeze(0).cpu().numpy()

            # 3. Return the calculated action
            return action.astype(np.float32)

        except Exception as e:
            print(f"RL Agent {self.id}: Error during model inference: {e}. Observation: {observation}. Returning default action.")
            # Return default action matching action_dim
            fallback_action_dim = self.action_dim if self.action_dim is not None else 2
            return np.array([0.0] * fallback_action_dim, dtype=np.float32)