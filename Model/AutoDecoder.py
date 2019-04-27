import tensorflow as tf
import numpy as np
from model import *
class AutoDecoder(model):
    def __init__(self, game_name, path, config, loss_name, direct = False):
        model.__init__(self, "Perceptron", game_name, config)
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.action_size = config.action_size
        self.direct = direct
        self.gamma = config.gamma
        self.decay = config.reg_decay
        self.learning_rate = config.reg_lr
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
        self._optimize()



    def _create_placeholders(self):
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="tf_x")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope('layer_one', reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W1 = tf.get_variable("W1", [self.feature_size, self.hidden_size], initializer=xavier_l1)
            with tf.variable_scope('layer_three', reuse=False):
                xavier_l3 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W3 = tf.get_variable("W2", [self.hidden_size, self.feature_size], initializer=xavier_l3)

    def _predict(self, x):
        h = tf.nn.relu(tf.matmul(x, self.W1))
        h = tf.matmul(h, self.W3)
        obs = tf.nn.relu(h)
        return obs

    def _loss_pre(self, loss_name):
        if loss_name == "MSE":
            mse = tf.nn.l2_loss(self.tf_x - self.pre_x)
            return tf.reduce_mean(mse)



    def train_model(self, data, action, reward, value, epoch, direct = False):
        features = self.feature_extract(data)
        feed = {self.tf_x: features}
        _, loss_rec = self.sess.run([self.train_op, self.loss], feed)
        if epoch % 10 == 0:
            print "epoch: {} rec loss: {}".format(epoch, loss_rec)

    def _optimize(self):
        self.pre_x = self._predict(self.tf_x)
        self.loss = self._loss_pre(self.loss_name)
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).minimize(self.loss)

    def init_model(self, path):
        save_dir = '/'.join(path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, load_path)

    def get_w(self):
        w1 = self.sess.run([self.W1])
        return w1[0]

    def save_model(self, path, episode_number):
        self.saver.save(self.sess, path, global_step=episode_number)
        print "SAVED MODEL #{}".format(episode_number)
