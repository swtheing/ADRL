import tensorflow as tf
import numpy as np
from model import *
class Perceptron(model):
    def __init__(self, game_name, path, config, loss_name, attribute = "reserve", reuse = False, copy_model = None, direct = False):
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.action_size = config.action_size
        self.direct = direct
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.decay = config.decay
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.batch_size = config.batch_size
        self.reuse = reuse
        print reuse
        if copy_model == None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = copy_model.sess
        self.var_list = []
        self.attri = attribute
        self._build_model()
        model.__init__(self, "Perceptron", game_name, config, copy_model)
        tf.initialize_all_variables().run()



    def feature_extract(self, data):
        if self.game_name == "Pong-v0":
            features = np.zeros([len(data), self.feature_size])
            for i in range(len(data)):
                feature = data[i]
                feature = feature[35:195]  # crop
                feature = feature[::2, ::2, 0]  # downsample by factor of 2
                feature[feature == 144] = 0  # erase background (background type 1)
                feature[feature == 109] = 0  # erase background (background type 2)
                feature[feature != 0] = 1  # everything else (paddles, ball) just set to 1
                feature = feature.astype(np.float).ravel()
                features[i,:] = feature
            return features
        return data

    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self._optimize_policy()

    def _get_params(self):
        value = self.sess.run([self.W1, self.W2])
        return value

    def _create_placeholders(self):
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="tf_x")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name="tf_action")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope(self.attri + '_reg_layer_one', reuse = self.reuse):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W1 = tf.get_variable("W1", [self.feature_size, self.hidden_size], initializer=xavier_l1)
            with tf.variable_scope(self.attri + '_reg_layer_two', reuse = self.reuse):
                xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.hidden_size), dtype=tf.float32)
                self.W2 = tf.get_variable("W2", [self.hidden_size, self.action_size], initializer=xavier_l2)
        self.var_list.append(self.W1)
        self.var_list.append(self.W2)

    def _assign(self, value):
        self.update_1 = self.W1.assign(value[0])
        self.update_2 = self.W2.assign(value[1])

    def _forward(self, x):  # x ~ [1,D]
        self.h = tf.matmul(x, self.W1)
        h = tf.nn.relu(self.h)
        logp = tf.matmul(h, self.W2)
        return logp

    def _loss_f(self, loss_name):
        if loss_name == "MSE":
            q = tf.multiply(self.logit, self.tf_action)
            q = tf.reduce_sum(q, axis=1)
            mse = tf.nn.l2_loss(self.tf_Q - q)
            return tf.reduce_mean(tf.multiply(mse, self.tf_epr))

        elif loss_name == "CE":
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.tf_action)
            return tf.reduce_sum(tf.multiply(cross_entropy, self.tf_epr))


    def test_model(self, data):
        features = self.feature_extract(data)
        feed = {self.tf_x : features}
        p, Q= self.sess.run([self.p, self.logit], feed)
        return p, Q


    def train_model(self, data, action, reward, value, epoch, direct = False):
        if self.direct:
            update_1, update_2 = self.sess.run([self.update_1, self.update_2])
        else:
            features = self.feature_extract(data)
            sparse_action = np.zeros((self.batch_size, self.action_size))
            for i in range(self.batch_size):
                sparse_action[i][action[i]] = 1
            feed = {self.tf_x : features, self.tf_action : sparse_action, self.tf_epr: reward, self.tf_Q: value}
            _, loss_val = self.sess.run([self.train_op, self.loss], feed)
            if epoch % 10 == 0:
                print "epoch: {} loss: {}".format(epoch, loss_val)


    def _optimize_policy(self):
        self.logit = self._forward(self.tf_x)
        self.p = tf.nn.softmax(self.logit)
        self.loss = self._loss_f(self.loss_name)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss)
        #self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).minimize(self.loss)
