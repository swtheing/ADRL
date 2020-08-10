import numpy as np

class env_con_toy(object):
    def __init__(self, name):
        self.name = name

    def reset(self):
        self.obs = np.random.rand(10)
        #self.obs = np.ones(10)
        return self.obs

    def step(self, action):
        self.obs = np.random.rand(10)
        #self.obs = np.ones(10)
        if np.sum(self.obs) < np.sum(action):
            reward = 1.0
        else:
            reward = -1.0
        return self.obs, reward, True, None
