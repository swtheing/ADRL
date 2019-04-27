from data_util import env
import gym
import numpy as np
#import cv2
import time
class env_atari(env):
    def __init__(self, name, config):
        env.__init__(self, name)
        self.env = gym.make(name)
        self.history_step = config.history_step
        self.ob_dims = config.ob_dims
        self.act_repeat = config.act_repeat
        self.history = []

    def reset(self):
        observation = self.env.reset()
        new_observation = self.get_observation(self.preprocess(observation))
        return new_observation

    def render(self):
        return self.env.render()

    def preprocess_old(self, observation):
        I = observation[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I

    def preprocess(self, observation):
        y = 0.2126 * observation[:, :, 0] + 0.7152 * observation[:, :, 1] + 0.0722 * observation[:, :, 2]
        y_screen = cv2.resize(y, (84,84))
        return y_screen

    def get_observation(self, obs):
        count = self.history_step
        self.history.append(obs)
        new_observation = np.zeros([self.ob_dims[0], self.ob_dims[1], self.history_step])
        for obs in reversed(self.history):
            count -= 1
            new_observation[:,:,count] = obs
            if count == 0:
                break
        while count != 0:
            count -= 1
            new_observation[:,:,count] = self.history[0]
        return new_observation

    def step(self, action):
        time.sleep(0.001)
        new_reward = 0
        for i in range(self.act_repeat):
            observation, reward, done, info = self.env.step(action)
            new_observation = self.get_observation(self.preprocess(observation))
            new_reward += reward
            new_info = info
            new_done = done
            if new_done:
                break

        return new_observation, new_reward, new_done, new_info
