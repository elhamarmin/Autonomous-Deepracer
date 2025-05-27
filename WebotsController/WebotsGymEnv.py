import gymnasium as gym
import numpy as np
from controller import Robot
from gymnasium import spaces

from WebotsEnvironment import WebotsEnvironment


class WebotsGymEnv(gym.Env):
    def __init__(self):
        super(WebotsGymEnv, self).__init__()
     
        # Initialize the Webots environment
        self.env = WebotsEnvironment()

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1,-1]),
                               high=np.array([1,1]),
                               dtype=np.float32)  # [0: forward, 1: right/left]
        
        shape = self.env.get_observations().shape
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        
    def reset(self,seed: int | None = None):
        """Resets the environment to its initial state and returns the initial observation."""
        return (self.env.reset(),{})

    def step(self, action):
        """Takes a step in the environment based on the action."""
        max_steps = 1000  # Define max steps per episode
        state, reward, done,truncated = self.env.step(action, max_steps)
        return state, reward, done, truncated, {}

    def render(self, mode='human'):
        """Renders the environment."""
        pass  # Implement render method if needed

    def close(self):
        """Performs any necessary cleanup."""
        pass  # Implement close method if needed
