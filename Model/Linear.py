import tensorflow as tf
import numpy as np
from model import *
class Linear(model):
    def __init__(self, game_name, path, config, loss_name, direct = False):
        model.__init__(self, "Perceptron", game_name, config)
        self.feature_size = config.feature_size
        self.action_size = config.action_size
        self.direct = direct
        self.gamma = config.gamma
        self.decay = config.decay
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.batch_size = config.batch_size
        self._build_model()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        if path != None:
            self.init_model(path)




    def feature_extract(self, data):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        features = np.zeros([len(data), self.feature_size])
        if self.game_name == "Pong-v0":
            for i in range(len(data)):
                feature = data[i]
                feature = feature[35:195]  # crop
                feature = feature[::2, ::2, 0]  # downsample by factor of 2
                feature[feature == 144] = 0  # erase background (background type 1)
                feature[feature == 109] = 0  # erase background (background type 2)
                feature[feature != 0] = 1  # everything else (paddles, ball) just set to 1
                feature = feature.astype(np.float).ravel()
                features[i,:] = feature
        else:
            for i in range(len(data)):
                features[i,:] = data[i]
        return features

    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self._optimize_policy()



    def _create_placeholders(self):
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="tf_x")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name="tf_action")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope('linear_reg_layer_one', reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W1 = tf.get_variable("W1", [self.feature_size, self.action_size], initializer=xavier_l1)

    def _assign(self, value):
        self.update_1 = tf.assign(self.W1, value[0])

    def _forward(self, x):  # x ~ [1,D]
        self.logit = tf.matmul(x, self.W1)
        return self.logit

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
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).minimize(self.loss)

    def init_model(self, path):
        save_dir = '/'.join(path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, load_path)


    def save_model(self, path, episode_number):
        self.saver.save(self.sess, path, global_step=episode_number)
        print "SAVED MODEL #{}".format(episode_number)

