#REINFORCE, MC, Policy Gradient
import random
import time
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.Linear import *
from Model.Cnn import *
from Model.Gaussian import *
from Model.Trans_Ptr import *
class MC_PG(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        if config.model == "DNN":
            Actor = Perceptron(game_name, None, config, "CE")
            self.conti_act = False
            self.multi_act = True
        elif config.model == "Gaussian":
            Actor = Gaussian(game_name, None, config, None)
            self.conti_act = True
            self.multi_act = True
        elif config.model == "CNN":
            Actor = Cnn(game_name, None, config, "CE")
            self.conti_act = False
            self.multi_act = True
        elif config.model == "TranPtr":
            Actor = Trans_Ptr(game_name, None, config, "CE")
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
        self.base_v = 0.0
        self.sum_step = 0
        self.viewer = None


    def Gen_Batch_Data(self, policy, epoch_num):
        batchs = []
        for epoch in range(epoch_num):
            samples = random.sample(range(len(self.replay_obs)), self.model.batch_size)
            samples_obs = [self.replay_obs[i] for i in samples]
            #bug
            #samples_act = [self.replay_act[i] - 1 for i in samples]
            samples_act = [self.replay_act[i] for i in samples]
            samples_next =[self.replay_next[i] for i in samples]
            samples_epr = []
            samples_Q = []
            for i in samples:
                if self.replay_done[i]:
                    #print self.replay_rew[i]
                    samples_epr.append(self.replay_rew[i])
                else:
                    for j in range(i+1, len(self.replay_obs)):
                        if self.replay_rew[j] != 0.0:
                            samples_epr.append(self.replay_rew[j])
                            break
                samples_Q.append(1.0)
            tup = (samples_obs, samples_act, samples_epr, samples_next)

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

        if not self.replay_switch:
            self.replay_obs = []
            self.replay_act = []
            self.replay_rew = []
            self.replay_done = []
            self.replay_next = []

        if len(self.replay_obs) == self.replay_size:
            del self.replay_obs[0]
            del self.replay_done[0]
            del self.replay_next[0]
            del self.replay_rew[0]
            del self.replay_act[0]

        self.replay_obs.append(observation)
        while True:
            action, Q, Q_debug = policy.action_sel(observation, max_sel = False, continues = self.conti_act, multi_act = self.multi_act)
            if len(self.replay_obs) > self.replay_size:
                del self.replay_obs[0]
                del self.replay_done[0]
                del self.replay_next[0]
                del self.replay_rew[0]
                del self.replay_act[0]
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
            self.replay_rew.append(reward)
            self.replay_done.append(done)
            self.replay_act.append(action)
            over_reward += reward
            match_rerward += reward
            if not done:
                self.replay_next.append(observation)
                self.replay_obs.append(observation)
            else:
                if match_rerward > max_reward:
                    max_reward = match_rerward
                elif match_rerward < min_reward:
                    min_reward = match_rerward
                match_rerward = 0
                self.replay_next.append(observation)
                match += 1
                if match == self.replay_match:
                    return over_reward / self.replay_match, max_reward, min_reward
                observation = self.env.reset()
                self.replay_obs.append(observation)
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
        samples_obs, samples_act, samples_epr, samples_next = train_data
        self.sum_step += 1
        mean_reward = np.mean(samples_epr)
        self.base_v = (self.base_v * (self.sum_step - 1) + mean_reward) / self.sum_step 
        print "base_v:"
        print self.base_v
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
            samples_epr -= self.base_v
            print "samples_epr:"
            print samples_epr
        policy.model.train_model(samples_obs, samples_act, samples_epr, samples_next, train_epoch)
