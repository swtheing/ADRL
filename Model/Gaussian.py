#Continue Action Space, Gaussian Action Prob, PG
import tensorflow as tf
import numpy as np
import math
import tensorflow_probability as tfp
from model import *

class Gaussian(model):
    def __init__(self, game_name, path, config, loss_name, direct = False):
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.batch_size = config.batch_size
        self.sess = tf.InteractiveSession()
        self._build_model()
        model.__init__(self, "Gaussian", game_name, config)
        tf.initialize_all_variables().run()


    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self._optimize_policy()

    def feature_extract(self, data):
        return data

    def _create_placeholders(self):
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="tf_x")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name="tf_action")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope('reg_layer_one', reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W1 = tf.get_variable("W1", [self.feature_size, self.action_size], initializer=xavier_l1)
            #with tf.variable_scope('reg_layer_two', reuse=False):
                #xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.hidden_size), dtype=tf.float32)
                #self.W2 = tf.get_variable("W2", [self.hidden_size, self.action_size], initializer=xavier_l2)
            with tf.variable_scope('reg_layer_three', reuse=False):
                xavier_l3 = tf.truncated_normal_initializer(mean=0, stddev = 1.0 / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W3 = tf.get_variable("W3", [self.feature_size, self.action_size * self.action_size], initializer=xavier_l3)
            #with tf.variable_scope('reg_layer_four', reuse=False):
                #xavier_l4 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.hidden_size), dtype=tf.float32)
                #self.W4 = tf.get_variable("W4", [self.hidden_size, self.action_size * self.action_size], initializer=xavier_l4)

    def _forward(self, x):
        mean_h = tf.matmul(x, self.W1)
        mean = mean_h
        #mean = tf.clip_by_value(mean_h, -100.0, 100.0)
        self.var_h = tf.matmul(x, self.W3)
        #var_h = tf.exp(self.var_h)
        var_h = tf.clip_by_value(self.var_h, -100.0, 100.0)
        var = tf.reshape(var_h, shape = [-1, self.action_size, self.action_size])
        #var_h = tf.reshape(var_h, shape = [-1, 1, self.action_size])
        #var = tf.matmul(var_h, var_h, transpose_a = [0, 2, 1])
        return mean, var

    def _Gaussian(self, mean, var):
        sub_mean = self.tf_action - mean
        numerator = tf.matmul(tf.transpose(sub_mean), tf.matrix_inverse(var))
        numerator = tf.matmul(numerator, sub_mean)
        numerator = tf.exp(-0.5 * numerator) 
        numerator = tf.matrix_diag(numerator)
        dimension = tf.cast(tf.shape(self.tf_x)[0], dtype=tf.float32)
        denominator = tf.sqrt(tf.pow(2 * math.pi, dimension) * tf.matrix_determinant(var))
        prob = numerator / denominator
        return prob

    def _Gen_prob(self, x):
        self.mean, self.var = self._forward(x)
        #self.mean = tf.Print(self.mean, [self.mean], summarize = 21)
        #act_prob = []
        #for i in range(self.batch_size):
        #    act_prob.append(self._Gaussian(self.mean[i,:], self.var[i, :, :]))
        #act_prob = tf.stack(act_prob)
        self.sample_act = tfp.distributions.MultivariateNormalTriL(loc=self.mean, scale_tril=self.var).sample([tf.shape(self.tf_x)[0]])
        log_act_prob = tfp.distributions.MultivariateNormalTriL(loc=self.mean, scale_tril=self.var).log_prob(self.tf_action)
        return log_act_prob

    def _optimize_policy(self):
        self.log_act_prob = self._Gen_prob(self.tf_x)
        self.loss = -self.log_act_prob * self.tf_epr
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss)


    def test_model(self, data):
        features = self.feature_extract(data)
        feed = {self.tf_x : features}
        act, var = self.sess.run([self.sample_act, self.var], feed)
        mean, var = self.sess.run([self.mean, self.var], feed)
        print mean, var 
        #act = np.random.multivariate_normal(mean[0], var[0])
        return act[0], None


    def train_model(self, data, action, reward, value, epoch, direct = False):
        features = self.feature_extract(data)
        feed = {self.tf_x : features, self.tf_action : action, self.tf_epr: reward, self.tf_Q: value}
        print reward
        _, loss_val, log_act_prob = self.sess.run([self.train_op, self.loss, self.log_act_prob], feed)
        print action
        print log_act_prob
        if epoch % 10 == 0:
            print "epoch: {} loss: {}".format(epoch, loss_val)

