import tensorflow as tf
import numpy as np
from model import *
np.set_printoptions(threshold='nan') 
class Cnn(model):
    def __init__(self, game_name, path, config, loss_name, attribute = "reserve", copy_model = None):
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.decay = config.decay
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.batch_size = config.batch_size
        self.attri = attribute
        self.var_list = []
        self._build_model()
        if copy_model == None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = copy_model.sess
        model.__init__(self, "Cnn", game_name, config, copy_model = copy_model)
        tf.initialize_all_variables().run()




    def feature_extract(self, data):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        #features = np.zeros([len(data), self.feature_size])
        #features = np.zeros([len(data), self.feature_size[0], self.feature_size[1], self.feature_size[2]])
        #for i in range(len(data)):
        #     feature = data[i]
        #     feature = feature[35:195]  # crop
        #     feature = feature[::2, ::2, 0]  # downsample by factor of 2
        #     feature[feature == 144] = 0  # erase background (background type 1)
        #     feature[feature == 109] = 0  # erase background (background type 2)
        #     feature[feature != 0] = 1  # everything else (paddles, ball) just set to 1
        #     feature = feature.astype(np.float).ravel()
        #     features[i, :] = np.reshape(feature, (self.feature_size[0], self.feature_size[1], self.feature_size[2]))
        #print features[0]
        #return features
        return data

    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self._optimize()


    def _create_placeholders(self):
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size[0], self.feature_size[1], self.feature_size[2]], name="tf_x")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name="tf_action")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="tf_keep_prob")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope(self.attri + '_layer_one', reuse=False):
                self.conv_filter_w1 = tf.get_variable("w1", shape=[2, 2, self.feature_size[2], 4], initializer= tf.contrib.layers.xavier_initializer())
                self.conv_filter_b1 = tf.get_variable("b1", shape=[4], initializer= tf.constant_initializer(0.1))
                self.conv_filter_w2 = tf.get_variable("w2", shape=[2, 2, 4, 8], initializer= tf.contrib.layers.xavier_initializer())
                self.conv_filter_b2 = tf.get_variable("b2", shape=[8], initializer= tf.constant_initializer(0.1))
                #self.conv_filter_w3 = tf.get_variable("w3", shape=[2,2,32,])
                # self.conv_filter_w1 = tf.Variable(tf.truncated_normal([8, 8, 3, 16], stddev= 0.1))
                # self.conv_filter_b1 = tf.Variable(tf.truncated_normal([16], stddev= 0.1))
                # self.conv_filter_w2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev= 0.01))
                # self.conv_filter_b2 = tf.Variable(tf.truncated_normal([32], stddev= 0.01))

            with tf.variable_scope(self.attri + '_FC', reuse=False):
                self.fc_w1 = tf.get_variable("fc_w1", shape=[32, self.hidden_size], initializer= tf.contrib.layers.xavier_initializer())
                self.fc_b1 = tf.get_variable("fc_b1", shape=[self.hidden_size], initializer= tf.constant_initializer(0.1))
                self.fc_w2 = tf.get_variable("fc_w2", shape=[self.hidden_size, self.action_size], initializer= tf.contrib.layers.xavier_initializer())
                self.fc_b2 = tf.get_variable("fc_b2", shape=[self.action_size], initializer= tf.constant_initializer(0.1))
            self.var_list.append(self.conv_filter_w1)
            self.var_list.append(self.conv_filter_b1)
            self.var_list.append(self.conv_filter_w2)
            self.var_list.append(self.conv_filter_b2)
            self.var_list.append(self.fc_w1)
            self.var_list.append(self.fc_b1)
            self.var_list.append(self.fc_w2)
            self.var_list.append(self.fc_b2)


    def _forward(self, x):  # x ~ [1,D]
        #x = tf.div(x, 255.)
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv_filter_w1, strides=[1,1,1,1], padding="VALID") + self.conv_filter_b1)#out: [210,160,5]
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.conv_filter_w2, strides=[1, 1, 1, 1], padding="VALID") + self.conv_filter_b2)
        flat = tf.reshape(conv2, [-1, 32])
        logit = tf.nn.relu(tf.matmul(flat, self.fc_w1) + self.fc_b1)
        logit = tf.nn.dropout(logit, keep_prob = self.keep_prob)
        logit = tf.matmul(logit, self.fc_w2) + self.fc_b2

        return logit

    def _loss_f(self, loss_name):
        if loss_name == "MSE":
            self.q = tf.multiply(self.logit, self.tf_action)
            self.q_sum = tf.reduce_sum(self.q, axis=1)
            mse = tf.square(self.tf_Q - self.q_sum)
            return tf.reduce_mean(tf.multiply(mse, self.tf_epr))

        elif loss_name == "CE":
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.tf_action)
            return tf.reduce_mean(tf.multiply(cross_entropy, self.tf_epr))
        elif loss_name == "CLP":
            self.q = tf.multiply(self.logit, self.tf_action)
            self.q_sum = tf.reduce_sum(self.q, axis=1)
            self.delta = self.tf_Q - self.q_sum
            self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                    0.5 * tf.square(self.delta),
                                    tf.abs(self.delta) - 0.5)
            return tf.reduce_mean(self.clipped_error)


    def test_model(self, data):
        features = self.feature_extract(data)
        feed = {self.tf_x : features, self.keep_prob: 1.0}
        p, Q= self.sess.run([self.p, self.logit], feed)
        return p, Q

    def train_model(self, data, action, reward, value, epoch, direct = False):
        features = self.feature_extract(data)
        sparse_action = np.zeros((self.batch_size, self.action_size))
        for i in range(self.batch_size):
            sparse_action[i][action[i]] = 1
        feed = {self.tf_x : features, self.tf_action : sparse_action, self.tf_epr: reward, self.tf_Q: value, self.keep_prob : 0.5}
        _, loss_val = self.sess.run([self.train_op, self.loss], feed)
        if epoch % 10 == 0:
            print "epoch: {} loss: {}".format(epoch, loss_val)#, q, q_sum)


    def _optimize(self):
        self.logit = self._forward(self.tf_x)
        self.p = tf.nn.softmax(self.logit)
        self.loss = self._loss_f(self.loss_name)
        #self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay, momentum=0.95, epsilon=0.01).minimize(self.loss)
        self.train_op =  tf.train.AdamOptimizer(self.learning_rate, epsilon = 0.000015).minimize(self.loss)
