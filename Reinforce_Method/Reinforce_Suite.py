import sys
import copy
from Policy import *

class Reinforce_Suite():
    def __init__(self, config, model, env):
        self.eval_epoch = config.eval_epoch
        self.epoch = config.epoch
        self.dis_epi = config.dis_epi
        self.train_epoch = config.train_epoch
        self.model = model
        self.policy_reserve = Policy(model, config.episilon)
        self.env = env
        self.gamma = config.gamma
        self.alpha = config.alpha
    def Get_Data(self, policy):
        raise NotImplementedError("Abstract Method")

    def Train_Data(self, policy_reserve, train_epoch, train_data):
        raise NotImplementedError("Abstract Method")

    def Policy_Evaluation(self, policy):
        mean_reward, max_reward, min_reward = self.Get_Data(policy)
        return mean_reward, max_reward, min_reward

    def Gen_Batch_Data(self, policy, epoch_num):
        raise NotImplementedError("Abstract Method")

    def Policy_Iteration(self):
        for iter in range(self.epoch):
            #Policy Evaluation
            reward = 0
            for epoch in range(self.eval_epoch):
                mean_reward, max_reward, min_reward = self.Policy_Evaluation(self.policy_reserve)
                print "epoch {}, eval max_reward: {}, min_reward: {}".format(epoch, max_reward, min_reward)
                reward += mean_reward
            print "iter {} eval reward: {}".format(iter, reward / self.eval_epoch)
            batches = self.Gen_Batch_Data(self.policy_reserve, self.train_epoch)
            for epoch in range(self.train_epoch):
                self.Train_Data(self.policy_reserve, epoch, batches[epoch])
            #Policy Improvement
            self.policy_reserve.expand_episilion(self.dis_epi)

    def Value_Iteration(self):
        raise NotImplementedError("Abstract Method")

    def Policy_Search(self):
        raise NotImplementedError("Abstract Method")







