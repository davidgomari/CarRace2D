# agents/random_agent.py
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, agent_id, action_space):
        super().__init__(agent_id)
        if action_space is None:
             raise ValueError("RandomAgent requires an action_space during initialization.")
        self.action_space = action_space # Gymnasium action space
        print(f"Random Agent {self.id} initialized with action space: {self.action_space}")

    def get_action(self, observation):
        return self.action_space.sample()