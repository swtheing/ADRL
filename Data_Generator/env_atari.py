from data_util import env
import gym
import numpy as np
import cv2
import time
class env_atari(env):
    def __init__(self, name, config):
        env.__init__(self, name)
        self.env = gym.make(name)
        self.history_step = config.history_step
        self.ob_dims = config.ob_dims

    def reset(self):
        observation = self.env.reset()
        new_observation = np.zeros([self.ob_dims[0], self.ob_dims[1], self.history_step])
        for i in range(self.history_step):
            new_observation[:,:,i] = self.preprocess(observation)
        return new_observation

    def render(self):
        return self.env.render()

    def preprocess(self, observation):
        I = observation[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I

    def preprocess_old(self, observation):
        y = 0.2126 * observation[:, :, 0] + 0.7152 * observation[:, :, 1] + 0.0722 * observation[:, :, 2]
        y_screen = cv2.resize(y, self.ob_dims)
        return y_screen

    def step(self, action):
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



