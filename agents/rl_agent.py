# agents/rl_agent.py
# Base agent and numpy import
from .base_agent import BaseAgent
import numpy as np
import os # Import os for path checking
# import torch # Example: Import PyTorch if needed

class RLAgent(BaseAgent):
    def __init__(self, agent_id, agent_info):
        super().__init__(agent_id)
        self.model = None
        self.model_path = agent_info.get('model_path')
        self.device = agent_info.get('device', 'cpu') # Example: Allow config to specify device

        if self.model_path:
            print(f"RL Agent {self.id}: Attempting to load model from {self.model_path}")
            if os.path.exists(self.model_path):
                try:
                    # --- Placeholder for your model loading logic --- 
                    # Example for PyTorch:
                    # self.model = YourPolicyNetwork(...) # Initialize network structure
                    # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    # self.model.to(self.device)
                    # self.model.eval() # Set model to evaluation mode
                    print(f"RL Agent {self.id}: Model structure/loading not implemented. Need to add PyTorch/TF logic here.")
                    # --- End Placeholder ---
                    print(f"RL Agent {self.id}: Model loading placeholder executed for path: {self.model_path}")
                except Exception as e:
                    print(f"RL Agent {self.id}: Failed to load model from {self.model_path}. Error: {e}")
                    self.model = None # Ensure model is None if loading fails
            else:
                print(f"RL Agent {self.id}: Model file not found at {self.model_path}. Agent will use default action.")
        else:
            print(f"RL Agent {self.id}: No model_path specified. Agent will use default action.")

    def get_action(self, observation):
        """ Gets action based on the observation dictionary. """
        if self.model:
            try:
                # --- Placeholder for your model inference logic --- 
                # Process the dictionary observation as needed by your model
                # Example: Stack specific components into a tensor
                # required_keys = ['x', 'y', 'v', 'theta'] # Example
                # if not all(key in observation for key in required_keys):
                #     print(f"RL Agent {self.id}: Missing required keys in observation: {observation.keys()}")
                #     return np.array([0.0, 0.0], dtype=np.float32)
                
                # obs_vector = np.concatenate([observation[key] for key in required_keys])
                # obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # with torch.no_grad():
                #     action_output = self.model(obs_tensor)
                # action = action_output.squeeze(0).cpu().numpy() 
                
                # --- Replace with actual inference --- 
                # Simple example: use velocity from dict
                speed = observation.get('v', np.array([0.0]))[0]
                # Output throttle/brake [-1, 1] and steer [-max_steer, +max_steer]
                # Example: Simple proportional brake based on speed, zero steer
                throttle_brake = -0.5 if speed > 5 else 0.1 # Apply some brake if fast
                steer = 0.0
                action_raw = np.array([throttle_brake, steer], dtype=np.float32) 
                # --- End Placeholder ---
                
                # Ensure action has exactly 2 elements before returning
                if action_raw.shape == (2,):
                     return action_raw
                else:
                     print(f"RL Agent {self.id} Error: Produced action with incorrect shape {action_raw.shape}. Returning default.")
                     return np.array([0.0, 0.0], dtype=np.float32)
                     
            except Exception as e:
                print(f"RL Agent {self.id}: Error during model inference: {e}. Observation: {observation}. Returning default action.")
                return np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Return a default action if model not loaded
            return np.array([0.0, 0.0], dtype=np.float32)