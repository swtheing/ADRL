import zoopt
import numpy as np
import random
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.AutoDecoder import *
from zoopt import Dimension, Objective, Parameter, Opt


class DPS(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        Reinforce_Suite.__init__(self, config, model, env)
        self.config = config
        self.game_name = game_name
        self.model = Perceptron(self.game_name, None, self.config, "MSE", direct = True)
        self.reg_model = AutoDecoder(self.game_name, None, self.config, "MSE")
        self.policy_reserve = Policy(self.model, self.config.episilon)
        self.replay_match = config.replay_match
        self.obs = []
        self.replay_size = config.replay_size
        self.pre = config.pre
        self.reg_epoch = config.reg_epoch

    def Pre_Reg(self, policy):
        if len(self.obs) == 0:
            mean_reward, max_reward, min_reward = self.Get_Data(self.policy_reserve)
        obs_batch = self.Gen_Batch_Data(self.policy_reserve, self.reg_epoch)
        for i in range(self.reg_epoch):
            self.reg_model.train_model(obs_batch[i], None, None, None, i)
        return self.reg_model.get_w()



    def Policy_Search(self):
        def ackley(solution):
            value = []
            if self.pre:
                w1 = self.Pre_Reg(self.policy_reserve)
            w_1_dim = (self.config.feature_size, self.config.hidden_size)
            w_2_dim = (self.config.hidden_size, self.config.action_size)
            w_flat = solution.get_x()
            if not self.pre:
                value.append(np.reshape(w_flat[0:w_1_dim[0] * w_1_dim[1]], w_1_dim))
                value.append(np.reshape(w_flat[w_1_dim[0] * w_1_dim[1]:], w_2_dim))
            else:
                value.append(w1)
                value.append(np.reshape(w_flat, w_2_dim))
            self.model._assign(value)
            self.model.train_model(None, None, None, None, None)
            mean_reward, max_reward, min_reward = self.Get_Data(self.policy_reserve)
            print "eval max_reward: {}, min_reward: {}, mean_reward: {}".format(max_reward, min_reward, mean_reward)
            return - mean_reward
        if not self.pre:
            dim = self.config.feature_size * self.config.hidden_size + self.config.hidden_size * self.config.action_size
        else:
            dim = self.config.hidden_size * self.config.action_size
        obj = Objective(ackley, Dimension(dim, [[-0.01, 0.01]] * dim, [True] * dim))
        solution = Opt.min(obj, Parameter(budget=100 * dim, uncertain_bits=100, intermediate_result=False, intermediate_freq=1))
        solution.print_solution()

    def Get_Data(self, policy):
        observation = self.env.reset()
        if len(self.obs) == self.replay_size:
            del self.obs[0]
        self.obs.append(observation)
        match = 0
        over_reward = 0
        max_reward = -21.0
        min_reward = 0
        match_rerward = 0.0
        observation_batch = [observation]
        while True:
            #self.env.render()
            action, Q, Q_debug = policy.action_sel(observation_batch)
            observation, reward, done, info = self.env.step(action)
            if len(self.obs) == self.replay_size:
                del self.obs[0]
            self.obs.append(observation)
            if not done:
                over_reward += reward
                match_rerward += reward
                observation_batch = [observation]
            else:
                if match_rerward > max_reward:
                    max_reward = match_rerward
                elif match_rerward < min_reward:
                    min_reward = match_rerward
                match_rerward = 0
                match += 1
                if match == self.replay_match:
                    return over_reward / self.replay_match, max_reward, min_reward
                observation = self.env.reset()
                observation_batch = [observation]

    def Gen_Batch_Data(self, policy, epoch_num):
        batchs = []
        for epoch in range(epoch_num):
            samples = random.sample(range(len(self.obs)), self.reg_model.batch_size)
            samples_obs = [self.obs[i] for i in samples]
            batchs.append(samples_obs)
        return batchs



