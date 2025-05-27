# Add the controller Webots Python library path
import datetime

from Interfaces.BaseEnvironment import BaseEnvironment
from Interfaces.BasePolicyNetwork import BasePolicyNetwork
from WebotsEnvironment import WebotsEnvironment

# Add Webots controlling libraries
from Policy_Network import Policy_Network

# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

import networkx as nx
import matplotlib.pyplot as plt
    
from torch.utils.tensorboard import SummaryWriter

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Agent_REINFORCE():
    """Agent implementing the REINFORCE algorithm."""

    def __init__(self,environment : BaseEnvironment, policy_network: BasePolicyNetwork ,save_path, load_path, num_episodes, max_steps, 
                  learning_rate, gamma, clip_grad_norm, baseline):
                
        self.save_path = save_path
        self.load_path = load_path
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Hyper-parameters Attributes
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learing_rate = learning_rate
        self.gamma = gamma
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline
        
        self.log_dir = "logs/train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir )
                
        # Initialize Network (Model)
        self.network = policy_network
    
        # Create the self.optimizers
        self.optimizer = optim.Adam(self.network.parameters(), self.learing_rate)
        
        # instance of env
        self.env = environment
               
        
    def save(self, path):
        """Save the trained model parameters after final episode and after receiving the best reward."""
        torch.save(self.network.state_dict(), self.save_path + path)
    
    
    def load(self):
        """Load pre-trained model parameters."""
        self.network.load_state_dict(torch.load(self.load_path))


    def compute_returns(self, rewards):
        """
        Compute the discounted returns.
        
        Parameters:
        - rewards (list): List of rewards obtained during an episode.
        
        Returns:
        - torch.Tensor: Computed returns.
        """

        # Generate time steps and calculate discount factors
        t_steps = torch.arange(len(rewards))
        discount_factors = torch.pow(self.gamma, t_steps).to(device)
    
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    
        # Calculate returns using discounted sum
        returns = rewards * discount_factors
        returns = returns.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,)) / discount_factors
    
        if self.baseline:
            mean_reward = torch.mean(rewards)
            returns -= mean_reward
        
        return returns

    
    def compute_loss(self, log_probs, returns):
        """
        Compute the REINFORCE loss.
        
        Parameters:
        - log_probs (list): List of log probabilities of actions taken during an episode.
        - returns (torch.Tensor): Computed returns for the episode.
        
        Returns:
        - torch.Tensor: Computed loss.
        """
            
        # Calculate loss for each time step
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
            
        # Sum the individual losses to get the total loss
        return torch.stack(loss).sum()

    
    def train(self):
        """
        Train the agent using the REINFORCE algorithm.
        
        This method performs the training of the agent using the REINFORCE algorithm. It iterates
        over episodes, collects experiences, computes returns, and updates the policy network.
        """           
    
        self.network.train()
        start_time = time.time()
        reward_history = []
        best_score = -np.inf
        
        for episode in range(1, self.num_episodes+1):
            done = False
            state = self.env.reset()
                
            log_probs = []
            rewards = []
            ep_reward = 0
            while True:
                action_probs = self.network(torch.as_tensor(state, device=device)) # action probabilities
                dist = torch.distributions.Categorical(action_probs) # Make categorical distrubation
                
                action = dist.sample() # Sample action
                log_prob = dist.log_prob(action) # The log probability of the action under the current policy distribution.
                log_probs.append(log_prob)
                
                my_action = action.item()
                next_state, reward, done = self.env.step(my_action, self.max_steps)
                
                rewards.append(reward)
                ep_reward += reward
                
                if done:
                    self.writer.add_histogram('Action Probabilities', action_probs.cpu().data.numpy(), episode)

                    self.writer.add_scalar('Episode Reward', ep_reward, episode)
                
    
                    returns = self.compute_returns(rewards)
                    loss = self.compute_loss(log_probs, returns)
                    
                    self.writer.add_scalar('Loss', loss.item(), episode)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    for name, param in self.network.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'{name}.grad', param.grad.cpu().data.numpy(), episode)


                    self.save(f'/ep_vis_{episode}')
                
                    # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.network.parameters(), float('inf'))
                    # print("Gradient norm before clipping:", grad_norm_before_clip)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    reward_history.append(ep_reward)
                            
                    # Log weights
                    for name, param in self.network.named_parameters():
                        self.writer.add_histogram(name, param.cpu().data.numpy(), episode)
                        
                    if ep_reward > best_score:
                        self.save(path='/best_weights.pt')
                        best_score = ep_reward
                    
                    print(f"Episode {episode}: Score = {ep_reward:.3f}")
                    break
                
                state = next_state
        
        # Save final weights and plot reward history
        self.save(path='/final_weights.pt')
        self.plot_rewards(reward_history)        
                
        # Print total training time
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
        
              
    def test(self):
        """
        Test the trained agent.
        This method evaluates the performance of the trained agent.
        """
        
        start_time = time.time()
        rewards = []
        self.load()
        self.network.eval()
        
        for episode in range(1, self.num_episodes+1):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action_probs = self.network(torch.as_tensor(state, device=device))
                action = torch.argmax(action_probs, dim=0)
                state, reward, done = self.env.step(action.item(), self.max_steps)
                ep_reward += reward
            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
        print(f"Mean Score = {np.mean(rewards):.3f}")
        
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
    
    
    def plot_rewards(self, rewards):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(rewards, np.ones(25)/25, mode='valid')
        
        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        plt.savefig(self.save_path + '/reward_plot.png', format='png', dpi=1000, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()
            
from WebotsGymEnv import WebotsGymEnv
from controller import Robot
            
def remove_files(directory):
    for file in os.listdir(directory):
        if file.startswith('ep_vis_'):
            os.remove(os.path.join(directory, file))
            print(f"Removed file: {file}")
            

if __name__ == '__main__':
    # Parameters
    save_path = './results'   
    load_path = './results/best_weights.pt'
    train_mode = True
    num_episodes = 2000 if train_mode else 10
    max_steps = 500 if train_mode else 100
    learning_rate = 1e-4
    gamma = 0.9
    hidden_size = 6
    clip_grad_norm = 5
    baseline = True

    remove_files(save_path)
    
    robot = Robot()
    
    my_env = WebotsGymEnv(robot)
    my_policy = Policy_Network(input_size=3, hidden_size=hidden_size, output_size=3).to(device)
    
    # Agent Instance
    agent = Agent_REINFORCE(policy_network=my_policy, environment=my_env, save_path=save_path, load_path=load_path, 
                            num_episodes=num_episodes,max_steps= max_steps, 
                            learning_rate=learning_rate,gamma= gamma,clip_grad_norm= clip_grad_norm,baseline= baseline)
    
    if train_mode:
        # Initialize Training
        agent.train()
    else:
        # Test
        agent.test()