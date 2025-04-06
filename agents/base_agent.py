# agents/base_agent.py
class BaseAgent:
    def __init__(self, agent_id):
        self.id = agent_id

    def get_action(self, observation):
        """ Returns an action [acceleration, steering_angle] based on observation. """
        raise NotImplementedError
