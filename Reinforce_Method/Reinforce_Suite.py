import sys
import copy
from Policy import *

class Reinforce_Suite():
    def __init__(self, config, model_reserve, env = None, model_predict = None):
        self.eval_epoch = config.eval_epoch
        self.epoch = config.epoch
        self.dis_epi = config.dis_epi
        self.train_epoch = config.train_epoch
        self.model = model_reserve
        self.policy_reserve = Policy(model_reserve, config.episilon)
        if model_predict != None:
            self.policy_predict = Policy(model_predict, config.episilon)
        else:
            self.policy_predict = self.policy_reserve
        self.env = env
        self.explore_iter = config.explore_iter
        self.save_path = config.save_path
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.min_epi = config.min_epi
        self.delay_update = config.delay_update

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
        iter_begin = 0
        #iter_begin = self.policy_reserve.model.restore_model(self.save_path)
        for iter in range(iter_begin, self.epoch):
            #Policy Evaluation
            if self.delay_update != 0 and iter % self.delay_update == 0:
                print "copy policy"
                self.policy_predict.copy_policy()
            #value = self.policy_predict.model.get_params()
            #print self.policy_reserve.model.sess.run(value[-1])
            #value = self.policy_reserve.model.get_params()
            #print self.policy_reserve.model.sess.run(value[-1])
            reward = 0
            for epoch in range(self.eval_epoch):
                mean_reward, max_reward, min_reward = self.Policy_Evaluation(self.policy_reserve)
                print "epoch {}, eval max_reward: {}, min_reward: {}".format(epoch, max_reward, min_reward)
                reward += mean_reward
            print "iter {} eval reward: {}".format(iter, reward / self.eval_epoch)
            if iter < self.explore_iter:
                continue
            batches = self.Gen_Batch_Data(self.policy_predict, self.train_epoch)
            for epoch in range(self.train_epoch):
                self.Train_Data(self.policy_reserve, epoch, batches[epoch])
            #W1, W2 = self.policy_predict.model._get_params()
            #print W2
            #W1, W2 = self.policy_reserve.model._get_params()
            #print W2
            #Todo: save_policy?
            #if iter % 50 == 0:
            #    self.policy_reserve.model.save_model(self.save_path, iter)
            #Policy Improvement
            self.policy_reserve.expand_episilion(self.dis_epi, self.min_epi)

    def Value_Iteration(self):
        raise NotImplementedError("Abstract Method")

    def Policy_Search(self):
        raise NotImplementedError("Abstract Method")
