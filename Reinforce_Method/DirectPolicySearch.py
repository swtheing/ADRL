import zoopt
import numpy as np
from Reinforce_Suite import *
from Perceptron import *
from zoopt import Dimension, Objective, Parameter, Opt


class DPS(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        Reinforce_Suite.__init__(self, config, model, env)
        self.config = config
        self.game_name = game_name
        self.model = Perceptron(self.game_name, None, self.config, "MSE")
        self.policy_reserve = Policy(self.model, self.config.episilon)
        self.replay_match = config.replay_match

    def Policy_Search(self):
        def ackley(solution):
            value = []
            w_1_dim = (self.config.feature_size, self.config.hidden_size)
            w_2_dim = (self.config.hidden_size, self.config.action_size)
            w_flat = solution.get_x()
            value.append(np.reshape(w_flat[0:w_1_dim[0] * w_1_dim[1]], w_1_dim))
            value.append(np.reshape(w_flat[w_1_dim[0] * w_1_dim[1]:], w_2_dim))
            self.model._assign(value)
            self.model.train_model(None, None, None, None, None, direct = True)
            mean_reward, max_reward, min_reward = self.Get_Data(self.policy_reserve)
            print "eval max_reward: {}, min_reward: {}, mean_reward: {}".format(max_reward, min_reward, mean_reward)
            return mean_reward
        dim = self.config.feature_size * self.config.hidden_size + self.config.hidden_size * self.config.action_size
        obj = Objective(ackley, Dimension(dim, [[-0.1, 0.1]] * dim, [True] * dim))
        solution = Opt.min(obj, Parameter(budget=100 * dim))
        solution.print_solution()
        import matplotlib.pyplot as plt
        plt.plot(obj.get_history_bestsofar())
        plt.savefig('figure.png')


    def Get_Data(self, policy):
        observation = self.env.reset()
        match = 0
        over_reward = 0
        max_reward = -21.0
        min_reward = 0
        match_rerward = 0.0
        observation_batch = [observation]
        while True:
            self.env.render()
            action, Q, Q_debug = policy.action_sel(observation_batch)
            observation, reward, done, info = self.env.step(action)
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

