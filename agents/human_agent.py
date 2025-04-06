# agents/human_agent.py
import pygame # Requires pygame installation
from .base_agent import BaseAgent
import numpy as np

class HumanAgent(BaseAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.throttle_brake = 0.0
        self.steer = 0.0
        print(f"Human Agent {self.id}: Use Arrow Keys (Up/Down=Throttle/Brake [-1, 1], Left/Right=Steer [-1, 1])")

    def get_action(self, observation):
        keys = pygame.key.get_pressed()
        self.throttle_brake = 0.0
        self.steer = 0.0

        if keys[pygame.K_UP]:
            self.throttle_brake = 1.0 # Max throttle
        if keys[pygame.K_DOWN]:
            self.throttle_brake = -1.0 # Max brake
        if keys[pygame.K_LEFT]:
            self.steer = 1.0 # Max left steer multiplier (positive angle)
        if keys[pygame.K_RIGHT]:
            self.steer = -1.0 # Max right steer multiplier (negative angle)

        # Action is [throttle_brake_input, steer_input]
        # Steer input will be scaled by max_steer_angle in car physics
        return np.array([self.throttle_brake, self.steer * np.pi/2], dtype=np.float32) # Multiply steer by pi/2 for approx range match