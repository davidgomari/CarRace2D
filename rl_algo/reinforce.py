import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np

class ReinforcePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_episodes=2000):
        super().__init__()
        self.max_episodes = max_episodes
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # It's better practice to pass the device during initialization or determine it when needed.
        # Using next(self.parameters()).device inside methods can be brittle if the model hasn't been moved yet.

    def forward(self, x):
        # Ensure input tensor is on the same device as the layer weights
        device = self.fc1.weight.device
        x = x.to(device) 
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        # Ensure log_std is on the same device
        log_std = self.log_std.to(device) 
        return mean, torch.exp(log_std)

    def act(self, state, episode):
        # Determine the device from the model parameters reliably
        model_device = self.fc1.weight.device
        
        eps = 0.999 ** (min(episode, self.max_episodes) * 2)
        # Use torch.rand on the model's device for the check
        if torch.rand(1, device=model_device).item() < eps:
            # pure exploration
            # Create exploration tensors on the correct device
            low  = torch.tensor([-1.0, -0.6], device=model_device) 
            high = torch.tensor([ 1.0,  0.6], device=model_device)
            # Ensure random tensor is also on the correct device
            action = (high - low) * torch.rand(2, device=model_device) + low 
            # dummy log_prob on the correct device
            log_prob = torch.zeros(1, device=model_device, requires_grad=True) 
            # Return action as numpy (needs cpu first), log_prob as tensor on model_device (requires grad)
            return action.detach().cpu().numpy(), log_prob # Return log_prob without detaching
        else:
            # policy network
            # Ensure state tensors are moved to the correct device before concatenation
            state_tensor = torch.cat([v.to(model_device) for v in state.values()], dim=-1)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            # Forward pass happens on model_device
            mean, std = self.forward(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample() # Action tensor is on model_device, shape (1, action_dim)
            log_prob = dist.log_prob(action).sum(dim=-1) # log_prob tensor is on model_device, shape (1,)
            
            # Return action as numpy (needs cpu first), log_prob as tensor on model_device (requires grad)
            # Squeeze action to remove batch dim before converting to numpy
            return action.squeeze(0).detach().cpu().numpy(), log_prob # Return log_prob without detaching