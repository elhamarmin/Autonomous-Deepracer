import datetime

import gymnasium as gym
import numpy as np
from controller import Robot
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback

import wandb
from WebotsEnvironment import WebotsEnvironment
from WebotsGymEnv import WebotsGymEnv


class EmptyENV(gym.Env):
    def __init__(self):
        super(EmptyENV, self).__init__()
     
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1,-1]),
                               high=np.array([1,1]),
                               dtype=np.float32)  # [0: forward, 1: right/left]
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(2,), dtype=np.float32)
        
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
    
    
class MacRacerTraining():
    def __init__(self,train=True) -> None:
        
        if train:
            env = WebotsGymEnv()
            check_env(env)
        else:
            env = EmptyENV()
    
        self.env = env
        
        self.sweep_configuration = {
            "method": "bayes",
            "metric": {"goal": "maximize", "name": "rollout/ep_rew_mean"},
            "parameters": {
                "gae_lambda": {"max": 0.95, "min": 0.80},
                "ent_coef": {"max": 0.25, "min": 0.1},
                "clip_range" : {"max": 0.15, "min": 0.05},
                "total_timesteps": {'value': 35000},
                'learning_rate': {
                    'values': [0.0001, 0.0002, 0.0003 ,0.0004]
                },
                'batch_size': {
                    'values': [128,256,512]
                },
                'n_epochs':{
                    'values': [16,32,64,128]
                },
                'n_steps': {
                    'values': [2048, 1024, 512, 256]
                }
            },
        }


        pass
    
    def load_model(self, model_path):
        model = PPO('MlpPolicy',env=self.env)
        model = model.load(model_path)
        return model
    
    def wandb_train(self):
        sweep_id = wandb.sweep(sweep=self.sweep_configuration, project="MacRacer")
        wandb.agent(sweep_id,project='MacRacer' ,function=self.train_model, count=30)
    
    def manual_train_model(self, learning_rate,ent_coef,gae_lambda,n_steps,batch_size,n_epochs,clip_range,total_timesteps):
    
        self.env.env.reset_rewards()
        
        model = PPO('MlpPolicy', self.env,learning_rate=learning_rate, 
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,tensorboard_log='./gym_logs/',
                    n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs, clip_range=clip_range, verbose=1,device='cuda')
        
        start_time = datetime.datetime.now()
        model = model.learn(total_timesteps=total_timesteps)
        
        end_time = datetime.datetime.now()
        
        print(f'Training done! Time: {(end_time - start_time).total_seconds()}\n Saving ... ')
        
        model.save(f'./trained_model/model_{end_time.strftime("%Y%m%d-%H%M")}.zip')
        
        return model

    def train_model(self):
        run = wandb.init(
            project="sb3",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        
        self.env.env.reset_rewards()
        self.env.env.set_log_writer(run)
        
        config = wandb.config
        
        model = PPO('MlpPolicy', self.env,learning_rate=config.learning_rate, 
                    ent_coef=config.ent_coef,
                    gae_lambda=config.gae_lambda,tensorboard_log='./gym_logs/',
                    n_steps=config.n_steps,batch_size=config.batch_size,n_epochs=config.n_epochs, clip_range=config.clip_range, verbose=1)
        
        model = model.learn(total_timesteps=config.total_timesteps,
                            callback=WandbCallback(
                                gradient_save_freq=100,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                            ))
        
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        model.save(f'./trained_model/model_{run.id}_{time}.zip')
        
        wandb.finish()
        
        return model