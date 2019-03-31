from model import *
import tensorflow as tf
import numpy as np
import random

class GAN_for_Policy(model):
    def __init__(self, game_name, path, config, loss_name, direct = False):
        model.__init__(self, "GAN", game_name, config)
        self.feature_size = config.feature_size * config.hidden_size + config.hidden_size * config.action_size
        self.noise_size = config.noise_size
        self.direct = direct
        self.gamma = config.gamma
        self.loss_name = loss_name
        self.batch_size = config.batch_size
        self._build_model()
        #with graph.as_default():
        #    self._build_model()
        #session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #session_config.gpu_options.allow_growth = True
        #self.sess = tf.Session(graph = graph, config=session_config) 
        self.sess = tf.InteractiveSession()
        self.w_samples = {}
        self.w_scores = {}
        self.saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        if path != None:
            self.init_model(path)


    def _noise_gen(self, number):
        z_sample_val = np.random.normal(0, 1, size=(number, self.noise_size)).astype(np.float32)
        return z_sample_val

    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self.loss, params = self._loss_f()
        self._optimize(self.loss, params)




    def _create_placeholders(self):
        self.tf_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_size], name="noise")
        self.tf_true = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="true")
        self.tf_label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope('layer_one', reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W1 = tf.get_variable("W1", [self.noise_size, self.feature_size], initializer=xavier_l1)
            with tf.variable_scope('layer_two', reuse=False):
                xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.feature_size), dtype=tf.float32)
                self.W2 = tf.get_variable("W2", [2 * self.feature_size, 1], initializer=xavier_l2)

    def _generator(self, x):
        output = tf.sigmoid(tf.matmul(x, self.W1)) - 0.5
        g_params = [self.W1]
        return output, g_params

    def _discriminator(self, input):
        d_params = [self.W2]
        return tf.matmul(input, self.W2), d_params

    def train_model(self, data, action, reward, value, epoch, direct = False):
        z_sample, tf_true, tf_label = data
        feed = {self.tf_noise : z_sample, self.tf_true : tf_true, self.tf_label : tf_label}
        _, _, loss_rec = self.sess.run([self.d_trainop, self.g_trainop, self.loss], feed)
        if epoch % 10 == 0:
            print "epoch: {} rec loss: {}".format(epoch, loss_rec)

    def _loss_f(self):
        self.W_fake, g_params = self._generator(self.tf_noise)
        W_true = tf.concat([self.W_fake, self.tf_true], axis= 1)
        ans, d_params = self._discriminator(W_true)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.tf_label, logits= ans)
        return loss, [g_params, d_params]

    def _optimize(self, loss, params):
        self.g_trainop = tf.train.AdamOptimizer(0.0001).minimize(-loss, var_list= params[0])
        self.d_trainop = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list= params[1])










