import zoopt
import numpy as np
import random
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.GAN_for_Policy import *


class Policy_Generator(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        Reinforce_Suite.__init__(self, config, model, env)
        self.config = config
        self.game_name = game_name
        self.model = Perceptron(self.game_name, None, self.config, "MSE", direct = True)
        self.policy_gen = GAN_for_Policy(self.game_name, None, self.config, "MSE")
        self.policy_reserve = Policy(self.model, self.config.episilon)
        self.replay_match = config.replay_match
        self.obs = []
        self.replay_size = config.replay_size
        self.pre = config.pre
        self.sample_size = config.sample_size
        self.epoch = config.epoch
        self.train_epoch = config.train_epoch

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

    def Policy_Search(self):
        for epoch in range(self.epoch):
            min_score, min_W, max_score, max_W, mean_score = self.Policy_Opt(epoch)
            print "epoch {}, min_socre: {}, max_score: {}, mean_score: {}".format(self.epoch, min_score, max_score, mean_score)
    
    def Policy_Opt(self, epoch):
        def ackley(w_flat):
            value = []
            w_1_dim = (self.config.feature_size, self.config.hidden_size)
            w_2_dim = (self.config.hidden_size, self.config.action_size)
            value.append(np.reshape(w_flat[0:w_1_dim[0] * w_1_dim[1]], w_1_dim))
            value.append(np.reshape(w_flat[w_1_dim[0] * w_1_dim[1]:], w_2_dim))
            self.model._assign(value)
            self.model.train_model(None, None, None, None, None)
            mean_reward, max_reward, min_reward = self.Get_Data(self.policy_reserve)
            print "eval max_reward: {}, min_reward: {}, mean_reward: {}".format(max_reward, min_reward, mean_reward)
            return mean_reward
        z_sample = self.policy_gen._noise_gen(self.sample_size)
        feed = {self.policy_gen.tf_noise: z_sample}
        W_fake = self.policy_gen.sess.run([self.policy_gen.W_fake], feed_dict=feed)
        count = len(self.policy_gen.w_samples.keys())
        min_score = 20.0
        max_score = -20.0
        mean_score = 0.0
        for i in range(self.sample_size):
            score = ackley(W_fake[0][i])
            if min_score > score:
                min_score = score
                min_W = W_fake[0][i]
            if max_score < score:
                max_score = score
                max_W = W_fake[0][i]
            mean_score += score / self.sample_size
            self.policy_gen.w_samples[count] = W_fake[0][i]
            self.policy_gen.w_scores[count] = score
            count += 1
        self.sort_sample = sorted(self.policy_gen.w_scores.items(), key=lambda x: x[1])
        pos_batches = []
        for i in range(self.policy_gen.batch_size):
            pos_batches.append(self.policy_gen.w_samples[self.sort_sample[i][0]])
        for iter in range(self.train_epoch):
            samples = random.sample(range(len(self.policy_gen.w_samples) - self.policy_gen.batch_size), self.policy_gen.batch_size)
            z_sample = self.policy_gen._noise_gen(2 * self.policy_gen.batch_size)
            neg_batches = []
            for i in samples:
                neg_batches.append(self.policy_gen.w_samples[self.sort_sample[i + self.policy_gen.batch_size][0]])
            pos_labels = np.ones(self.policy_gen.batch_size)
            neg_labels = np.zeros(self.policy_gen.batch_size)
            labels = np.reshape(np.concatenate([pos_labels, neg_labels], axis=0), [2 * self.policy_gen.batch_size, -1])
            data = (z_sample, pos_batches + neg_batches, labels)
            self.policy_gen.train_model(data, None, None, None, iter)

        return min_score, min_W, max_score, max_W, mean_score






