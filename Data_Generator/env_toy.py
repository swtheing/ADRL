import gym
import numpy as np
from data_util import env
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

class env_toy(env):
    def __init__(self, preprocess):
        self.env = gym.make("FrozenLake-v0")
        self.preprocess = preprocess
    def reset(self):
        obs = self.env.reset()
        self.obs = obs
        if self.preprocess == "CNN":
            self.out = self.preprocess_cnn(obs)
        elif self.preprocess == "DNN":
            self.out = self.preprocess_dnn(obs)
        elif self.preprocess == "RAW":
            self.out = self.preprocess_raw(obs)
        return self.out
    
    def preprocess_dnn(self, obs):
        s = np.zeros(shape = (16,))
        s[obs] = 1
        return s
    
    def preprocess_raw(self, obs):
        obs += 1
        return [obs]

    def preprocess_cnn(self, obs):
        s = np.zeros(shape = (16,))
        s[obs] = 1
        return np.reshape(s, (4,4,1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action - 1)
        #if obs == self.obs:
        #    done = True
        if (reward == 0 and done):
            reward = -1
        self.obs = obs
        #print obs
        #print reward
        if self.preprocess == "CNN":
            self.out = self.preprocess_cnn(obs)
        elif self.preprocess == "DNN":
            self.out = self.preprocess_dnn(obs)
        elif self.preprocess == "RAW":
            self.out = self.preprocess_raw(obs)
        return self.out, reward, done, info
            
