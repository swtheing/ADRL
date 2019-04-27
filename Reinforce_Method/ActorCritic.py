#Actor-Critic On-Policy, TD(0)
import random
import time
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.Linear import *
class ActorCritic(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        Actor = Perceptron(game_name, None, config, "CE", attribute = "Actor")
        Reinforce_Suite.__init__(self, config, Actor, env)
        self.Critic = Perceptron(game_name, None, config, "MSE", attribute = "Critic")
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
        self.viewer = None


    def Gen_Batch_Data(self, policy, epoch_num):
        batchs = []
        for epoch in range(epoch_num):
            samples = random.sample(range(len(self.replay_obs)), self.model.batch_size)
            samples_obs = [self.replay_obs[i] for i in samples]
            samples_act = [self.replay_act[i] - 1 for i in samples]
            samples_epr = []
            samples_Q = []
            for i in samples:
                if self.replay_done[i]:
                    samples_Q.append(self.replay_rew[i])
                    samples_epr.append(self.replay_rew[i])
                else:
                    p, Q = self.Critic.test_model([self.replay_obs[i]])
                    if self.on_policy:
                        samples_Q.append(self.replay_rew[i] + self.gamma * Q[0, self.replay_act[i+1] - 1])
                    else:
                        samples_Q.append(self.replay_rew[i] + self.gamma * np.max(Q))
                    samples_epr.append(Q[0, self.replay_act[i] - 1])
            #print samples_Q
            tup = (samples_obs, samples_act, samples_epr, samples_Q)

            batchs.append(tup)
        return batchs

    def Get_Data(self, policy):
        observation = self.env.reset()
        match = 0
        over_reward = 0
        max_reward = -21.0
        min_reward = 0
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
            action, Q, Q_debug = policy.action_sel(observation, max_sel= False)
            if len(self.replay_obs) > self.replay_size:
                del self.replay_obs[0]
                del self.replay_done[0]
                del self.replay_next[0]
                del self.replay_rew[0]
                del self.replay_act[0]
            #self.env.render()
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
            observation, reward, done, info = self.env.step(action)
            self.replay_rew.append(reward)
            self.replay_done.append(done)
            self.replay_act.append(action)
            if not done:
                over_reward += reward
                match_rerward += reward
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

    def Train_Data(self, policy, train_epoch, train_data):
        #samples = self.random_sampling()
        #print [self.replay_Q[i] for i in samples]
        #print "sample ok"
        #print len(self.replay_obs)
        #print len(self.replay_act)
        #print len(self.replay_rew)
        #print len(self.replay_next)
        #print self.replay_Q
        samples_obs, samples_act, samples_epr, samples_Q = train_data
        self.Train_Critic(train_data, train_epoch)
        #print samples_epr
        policy.model.train_model(samples_obs, samples_act, samples_epr, samples_Q, train_epoch)

    def Train_Critic(self, train_data, train_epoch):
        #print "Train Critic"
        samples_obs, samples_act, samples_epr, samples_Q = train_data
        #print samples_Q
        samples_epr = [1.0 for i in range(len(samples_epr))]
        self.Critic.train_model(samples_obs, samples_act, samples_epr, samples_Q, train_epoch)
