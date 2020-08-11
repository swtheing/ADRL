import random
import time
from Reinforce_Suite import *
#from Perceptron import *
from Cnn import *
from gym.envs.classic_control import rendering

class MC_RL(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        model = Cnn(game_name, None, config, "MSE")
        Reinforce_Suite.__init__(self, config, model, env)
        self.replay_match = config.replay_match
        self.replay_size = config.replay_size
        self.observe_id = config.observe_id
        self.replay_obs_act = []
        self.replay_obs_num = []
        self.replay_Q = []
        self.replay_epr = []
        self.viewer = None

    def update_Q(self, replay_obs_act_set, G):
        for obs_act in range(replay_obs_act_set):
            if obs_act in self.replay_obs_act:
                id = self.replay_obs_act.index(obs_act)
                self.replay_Q[id] = self.replay_Q[id] + 1.0 / (self.replay_obs_num + 1) * (G - self.replay_Q[id])
                self.replay_obs_num[id] += 1
            elif len(self.replay_obs_act) == self.replay_size:
                del self.replay_obs_act[0]
                del self.replay_Q[0]
                del self.replay_obs_num[0]
                del self.replay_epr[0]
                self.replay_obs_act.append(obs_act)
                self.replay_Q.append(G)
                self.replay_obs_num.append(1)
                self.replay_epr.append(1.0)
            else:
                self.replay_obs_act.append(obs_act)
                self.replay_Q.append(G)
                self.replay_obs_num.append(1)
                self.replay_epr.append(1.0)

    def Cal_G(self, replay_reward):
        G = []
        for i in range(len(replay_reward)):
            G.append(0.0)
            for j in range(len(replay_reward)-1, -1, -1):
                G[i] += self.gamma * G[i] + self.gamma * replay_reward[j]
        return G

    def Gen_Batch_Data(self, policy, epoch_num):
        batchs = []
        for epoch in range(epoch_num):
            samples = random.sample(range(len(self.replay_obs_act)), self.model.batch_size)
            samples_obs = [self.replay_obs_act[i][0] for i in samples]
            samples_act = [self.replay_obs_act[i][1] - 1 for i in samples]
            samples_epr = [1.0 for i in samples]
            samples_Q = [self.replay_Q[i] for i in samples]
            #print samples_Q
            tup = (samples_obs, samples_act, samples_epr, samples_Q)
            batchs.append(tup)
        return batchs



    def Get_Data(self, policy):
        observation = self.env.reset()
        match = 0
        over_reward = 0
        replay_obs_set = []
        replay_reward = []
        while True:
            over_reward += reward
            action, Q, Q_debug = policy.action_sel(observation, max_sel = True, continues = self.conti_act, multi_act = self.multi_act)
            replay_obs_set.append((observation, action))
            if reward != 0:
                G = self.Cal_G(replay_reward)
                self.update_Q(replay_obs_set, G)
                replay_obs_set = []
            if done:
                match += 1
                if match == self.replay_match:
                    return over_reward / self.replay_match
                G = self.Cal_G(replay_reward)
                self.update_Q(replay_obs_set, G)
                observation = self.env.reset()
                replay_obs_set = []
                action, Q, Q_debug = policy.action_sel(observation, max_sel = True, continues = self.conti_act, multi_act = self.multi_act)
                replay_obs_set.append((observation, action))
                replay_reward.append(reward)


    def Train_Data(self, policy, train_epoch, train_data):
        #samples = self.random_sampling()
        #print [self.replay_Q[i] for i in samples]
        #print "sample ok"
        #print len(self.replay_obs)
        #print len(self.replay_act)
        #print len(self.replay_rew)
        #print len(self.replay_Q)
        #print self.replay_Q

        samples_obs, samples_act, samples_epr, samples_Q = train_data
        policy.model.train_model(samples_obs, samples_act, samples_epr, samples_Q, train_epoch)
        return policy
