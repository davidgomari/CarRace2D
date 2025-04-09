import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np

class ReinforcePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Use a parameter for log standard deviation (same for all states)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def act(self, state):
        # Handle dictionary observations by converting to numpy array
        if isinstance(state, dict):
            # Concatenate all observation components into a single numpy array
            state_array = np.concatenate([np.array(state[key]) for key in state.keys()])
        else:
            state_array = state
            
        state = torch.from_numpy(state_array).float().unsqueeze(0)  # add batch dim
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # For training, return the tensor (do not detach or call .item() on log_prob)
        return action.detach().numpy()[0], log_prob