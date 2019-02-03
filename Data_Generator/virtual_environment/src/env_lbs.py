#from data_util import env
#import gym
import numpy as np
#import cv2
import time

class EnvLBS():
#class env_lbs(env):
    def __init__(self, name, config):
        #env.__init__(self, name)
        #self.env = gym.make(name)
        self.history_step = config.history_step
        self.ob_dims = config.ob_dims
        self.observation = []

    def reset(self):
        """
        clear all state
        """
        observation = self.env.reset()
        new_observation = np.zeros([self.ob_dims[0], self.ob_dims[1], self.history_step])
        for i in range(self.history_step):
            new_observation[:,:,i] = self.preprocess(observation)
        return new_observation

    def render(self):
        return self.env.render()

    def preprocess(self, observation):
        """
        init state
        """

        return observation

    def step(self, action):
        """
        input action
        process
        output state
        """
        return observation



    def preprocess_old(self, observation):
        """
        new observation
        init state
        """
        I = observation[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I

    def step_old(self, action):
        new_observation = np.zeros([self.ob_dims[0], self.ob_dims[1], self.history_step])
        new_done = False
        new_reward = 0
        new_info = None
        done_pos = 0
        for i in range(self.history_step):
            if new_done:
                new_observation[:,:,i] = new_observation[:,:,done_pos]
            else:
                time.sleep(0.01)
                observation, reward, done, info = self.env.step(action)
                new_observation[:, :, i] = self.preprocess(observation)
                new_reward = reward
                new_info = info
                if done:
                    new_done = True
                    done_pos = i

        return new_observation, new_reward, new_done, new_info



