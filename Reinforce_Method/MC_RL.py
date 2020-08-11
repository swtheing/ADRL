#REINFORCE, MC, Policy Gradient
import random
import time
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.Linear import *
from Model.Cnn import *
from Model.Gaussian import *
from Model.Trans_Ptr import *
class MC_Q(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        if config.model == "DNN":
            Actor = Perceptron(game_name, None, config, "MSE")
            self.conti_act = False
            self.multi_act = False
        elif config.model == "Gaussian":
            Actor = Gaussian(game_name, None, config, None)
            self.conti_act = True
            self.multi_act = False
        elif config.model == "CNN":
            Actor = Cnn(game_name, None, config, "MSE")
            self.conti_act = False
            self.multi_act = False
        elif config.model == "TranPtr":
            Actor = Trans_Ptr(game_name, None, config, "MSE")
            self.conti_act = False
            self.multi_act = True
        Reinforce_Suite.__init__(self, config, Actor, env)
        self.replay_match = config.replay_match
        self.replay_size = config.replay_size
        self.observe_id = config.observe_id
        self.on_policy = config.on_policy
        self.replay_switch = config.replay_switch
        self.replay_obs = []
        self.replay_act = []
        self.replay_rew = []
        self.replay_done = []
        self.replay_next = []
        self.replay_Q = []
        self.base_v = 0.0
        self.sum_step = 0
        self.viewer = None


    def update_Q(self, replay_obs_act, G):
        for obs_act in range(replay_obs_act):
            if obs_act in self.replay_obs:
                id = self.replay_obs_act.index(obs_act)
                self.replay_Q[id] = self.replay_Q[id] * (1.0 - self.alpha) + self.alpha * G
                #self.replay_Q[id] = self.replay_Q[id] + 1.0 / (self.replay_obs_num + 1) * (G - self.replay_Q[id])
                #self.replay_obs_num[id] += 1
            elif len(self.replay_obs_act) == self.replay_size:
                del self.replay_obs_act[0]
                del self.replay_Q[0]
                del self.replay_obs_num[0]
                del self.replay_epr[0]
                self.replay_obs.append(obs_act)
                self.replay_Q.append(G)
                
                #self.replay_obs_num.append(1)
                self.replay_epr.append(1.0)
            else:
                self.replay_obs.append(obs_act)
                self.replay_Q.append(G)
                #self.replay_obs_num.append(1)
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
            samples = random.sample(range(len(self.replay_obs)), self.model.batch_size)
            samples_obs = [self.replay_obs[i] for i in samples]
            #bug
            #samples_act = [self.replay_act[i] - 1 for i in samples]
            samples_act = [self.replay_act[i] for i in samples]
            samples_next =[self.replay_next[i] for i in samples]
            samples_epr = [1.0 for i in samples]
            samples_Q = []
            for i in samples:
                if self.replay_done[i]:
                    samples_Q.append(self.replay_rew[i])
                else:
                    for j in range(i, len(self.replay_obs)):
                        if self.replay_rew[j] != 0.0:
                            samples_Q.append(self.replay_rew[j])
                            break
            #print "Samples_Q:"
            #print samples_Q
            tup = (samples_obs, samples_act, samples_epr, samples_Q, samples_next)

            batchs.append(tup)
        return batchs

    def Get_Data(self, policy):
        observation = self.env.reset()
        match = 0
        over_reward = 0
        max_reward = -1000000.0
        min_reward = 1000000.0
        match_rerward = 0.0
        show_flag = 1
        replay_obs = []
        replay_act = []
        replay_done = []
        replay_rew = []
        replay_obs.append(observation)
        while True:
            action, Q, Q_debug = policy.action_sel(observation, max_sel = True, continues = self.conti_act, multi_act = self.multi_act)
            #replay strategy
            # if self.observe_id < len(self.replay_obs):
            #     self.observe_picture = self.replay_obs[self.observe_id][25:,:,:]
            #     if (observation[25:,:,:] == self.observe_picture).all():
            #         if self.viewer is None:
            #             self.viewer = rendering.SimpleImageViewer()
            #         if show_flag == 1:
            #             self.viewer.imshow(observation[25:,:,:])
            #             show_flag = 0
            #         print "observe id: {}, action: {}, Q: {}".format(self.observe_id, action, Q_debug)
                #raw_input("trace image is here (Enter go): ");
            #action = [self.Greedy_action(observation) + 1]
            observation, reward, done, info = self.env.step(action)
            replay_rew.append(reward)
            replay_done.append(done)
            replay_act.append(action)
            over_reward += reward
            match_rerward += reward
            if not done:
                replay_next.append(observation)
                replay_obs.append(observation)
            else:
                if match_rerward > max_reward:
                    max_reward = match_rerward
                elif match_rerward < min_reward:
                    min_reward = match_rerward
                match_rerward = 0
                replay_next.append(observation)
                match += 1
                if match == self.replay_match:
                    return over_reward / self.replay_match, max_reward, min_reward
                observation = self.env.reset()
                replay_obs.append(observation)
    
    def distance(self, gps_1, gps_2):
        return pow((pow((gps_1[0] - gps_2[0]), 2) + pow((gps_1[1] - gps_2[1]), 2)), 0.5)
    
    def Greedy_action(self, obs):
        ans = []
        for i in range(obs[0].shape[0]):
            gps_1 = obs[0][i,:]
            gps_2 = obs[1][0,:]
            ans.append(self.distance(gps_1, gps_2))
        act = np.argmax(ans)
        return act

    def Train_Data(self, policy, train_epoch, train_data, rescale = True):
        #samples = self.random_sampling()
        #print [self.replay_Q[i] for i in samples]
        #print "sample ok"
        #print len(self.replay_obs)
        #print len(self.replay_rew)
        #print len(self.replay_next)
        #print self.replay_Q
        samples_obs, samples_act, samples_epr, samples_Q, samples_next = train_data
        if rescale:
            #for i in range(len(samples_epr)):
            #    samples_epr[i] = 1.0
            #    samples_act[i][0] = self.Greedy_action(samples_obs[i])
            #max_id = np.argmax(samples_epr)
            #min_id = np.argmin(samples_epr)
            #for i in range(len(samples_epr)):
            #    samples_epr[i] = 0.0
            #samples_epr[max_id] = 1.0
            #samples_epr[min_id] = -1.0
            #for i in range(len(samples_act)):
            #    for j in range(len(samples_act[i])):
            #        samples_act[i][j] = 5
            #samples_epr[0] = 1.0
            print "samples_epr:"
            print samples_epr
        policy.model.train_model(samples_obs, samples_act, samples_epr, samples_Q, samples_next, train_epoch)
