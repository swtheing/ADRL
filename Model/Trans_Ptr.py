import tensorflow as tf
import numpy as np
from model import *
from T2T import *
np.set_printoptions(threshold='nan') 
class Trans_Ptr(model):
    def __init__(self, game_name, path, config, loss_name, attribute = "reserve", copy_model = None):
        self.max_par_size = config.max_par_size
        self.max_task_size = config.max_task_size
        self.par_feature_size = config.par_feature_size
        self.task_feature_size = config.task_feature_size
        self.num_block = config.num_block
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.max_par_size = config.max_par_size
        self.action_size = config.max_par_size * config.max_task_size
        self.gamma = config.gamma
        self.decay = config.decay
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.dropout_prob = config.dropout_prob
        self.batch_size = config.batch_size
        self.attri = attribute
        self.initial_lstm_state()
        self.var_list = []
        self._build_model()
        if copy_model == None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = copy_model.sess
        model.__init__(self, "Trans_tr", game_name, config, copy_model = copy_model)
        tf.initialize_all_variables().run()




    def feature_extract(self, data, next_obs = None, rescale = True):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        task_fea = np.zeros([len(data), self.max_task_size, self.task_feature_size])
        par_fea = np.zeros([len(data), self.max_par_size, self.par_feature_size])
        task_all = np.zeros([len(data), self.max_task_size, self.task_feature_size])
        for i in range(len(data)):
            par_fea[i,:,:] = data[i][0]
            task_fea[i,:,:] = data[i][1]
            task_all[i,:,:] = data[i][1]
            if next_obs != None:
                count = 0
                for j in range(task_all.shape[2]):
                    if np.mean(task_all[i,j,:]) == 0:
                        task_all[i,j,:] = next_obs[i][1][count,:]
                        count += 1
        #if rescale:
        #    print "shape"
        #    par_mean = np.mean(par_fea, axis = 0, keepdims = True)
        #    par_std = np.std(par_fea, axis = 0, keepdims = True)
        #    task_mean = np.mean(task_fea, axis = 0, keepdims = True)
        #    task_std = np.std(task_fea, axis = 0, keepdims = True)
        #    print par_mean
        #    print par_std
        #    par_fea = (par_fea - par_mean) / par_std
        #    task_fea = (task_fea - task_mean) / task_std
        
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
        return par_fea, task_fea, task_all

    def _build_model(self):
        self._create_placeholders()
        self._create_w()
        self._optimize()


    def _create_placeholders(self):
        self.tf_par = tf.placeholder(dtype=tf.float32, shape=[None, self.max_par_size, self.par_feature_size], name="tf_par")
        self.tf_task = tf.placeholder(dtype=tf.float32, shape=[None, self.max_task_size, self.task_feature_size], name="tf_task")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.max_par_size], name="tf_action")
        self.initial_state_c = tf.placeholder(dtype=tf.float32, shape=[None, self.max_par_size], name="tf_state_c")
        self.initial_state_h = tf.placeholder(dtype=tf.float32, shape=[None, self.max_par_size], name="tf_state_h")
        
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.task_all = tf.placeholder(dtype=tf.float32, shape=[None, self.max_task_size, self.task_feature_size], name="task_all")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="tf_keep_prob")

    def _create_w(self):
        with tf.name_scope("data"):
            with tf.variable_scope(self.attri + '_FC', reuse=False):
                self.fc_w1 = tf.get_variable("fc_w1", shape=[32, self.hidden_size], initializer= tf.contrib.layers.xavier_initializer())
                self.fc_b1 = tf.get_variable("fc_b1", shape=[self.hidden_size], initializer= tf.constant_initializer(0.1))
                self.fc_w2 = tf.get_variable("fc_w2", shape=[self.hidden_size, self.action_size], initializer= tf.contrib.layers.xavier_initializer())
                self.fc_b2 = tf.get_variable("fc_b2", shape=[self.action_size], initializer= tf.constant_initializer(0.1))

    def _predict_task(self, tf_predict):
        self.task_predict = tf.layers.dense(self.dec, self.task_feature_size)
        mse = tf.square(tf_predict - self.task_predict)
        return tf.reduce_mean(mse)
    
    def _forward(self, tf_par, tf_task):  # x ~ [1,D]
        tf_par_norm = tf.layers.batch_normalization(tf_par, training = self.is_training)
        tf_par_norm = tf.Print(tf_par_norm, [tf_par_norm], summarize = 20, message = "tf_par")
        tf_task_norm = tf.layers.batch_normalization(tf_task, training = self.is_training)
        tf_task_norm = tf.Print(tf_task_norm, [tf_task_norm], summarize = 20, message = "tf_task")
        par_emb = tf.layers.dense(tf_par_norm, self.hidden_size)
        #par_emb = tf.Print(par_emb, [par_emb], summarize = 20, message = "par_emb")
        task_emb = tf.layers.dense(tf_task_norm, self.hidden_size)
        #task_emb = tf.keras.layers.LSTM(self.hidden_size, return_sequences = True)(task_emb)
        self.T2T = T2T_Model(None, self.hidden_size, self.dropout_prob, self.num_block, self.is_training, self.num_heads, self.hidden_size)
        self.dec, pointer = self.T2T.Trans_Pointer(par_emb, task_emb, emb = False)
        print pointer.shape.as_list()
        print self.initial_state_c.shape.as_list()
        print self.initial_state_h.shape.as_list()
        pointer, lstm_c, lstm_h = tf.keras.layers.LSTM(self.max_par_size, return_sequences = True, return_state = True)(pointer, initial_state = [self.initial_state_c, self.initial_state_h])
        print pointer.shape.as_list()
        self.state = [lstm_c, lstm_h]
        logit = tf.reshape(pointer, shape = (-1, self.max_par_size))
        return logit

    def _loss_f(self, loss_name):
        ##No MSE 
        if loss_name == "MSE":
            self.q = tf.multiply(self.logit, self.tf_action)
            self.q_sum = tf.reduce_sum(self.q, axis=1)
            mse = tf.square(self.tf_Q - self.q_sum)
            return tf.reduce_mean(tf.multiply(mse, self.tf_epr))
        elif loss_name == "CE":
            #??
            #tf_epr = (tf.sigmoid(self.tf_epr) - 0.5) * 2
            #tf_epr = tf.Print(tf_epr, [tf_epr], summarize = 20, message = "tf_epr")
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.tf_action)
            cross_entropy = tf.reshape(cross_entropy, shape = [-1, self.max_task_size])
            cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_1")
            mask_task = tf.reduce_sum(self.tf_task, axis = -1)
            mask_task = tf.Print(mask_task, [mask_task], summarize = 20, message = "mask_task")
            paddings = tf.zeros_like(cross_entropy)
            cross_entropy = tf.where(tf.equal(mask_task, 0), paddings, cross_entropy)
            cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_2")
            cross_entropy = tf.reduce_sum(cross_entropy, axis = -1)
            cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_3")
            return tf.reduce_mean(cross_entropy * self.tf_epr)
        elif loss_name == "CLP":
            self.q = tf.multiply(self.logit, self.tf_action)
            self.q_sum = tf.reduce_sum(self.q, axis=1)
            self.delta = self.tf_Q - self.q_sum
            self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                    0.5 * tf.square(self.delta),
                                    tf.abs(self.delta) - 0.5)
            return tf.reduce_mean(self.clipped_error)


    def test_model(self, data):
        par_fea, task_fea, task_all = self.feature_extract(data)
        self.initial_lstm_state(1)
        print "feature:"
        print task_fea
        print par_fea
        feed = {self.tf_par : par_fea, self.tf_task : task_fea, self.initial_state_c: self.lstm_state[0], self.initial_state_h: self.lstm_state[1], self.keep_prob: 1.0, self.is_training: False}
        p, Q, self.lstm_state = self.sess.run([self.p, self.logit, self.state], feed)
        print self.lstm_state[0].shape
        return p, Q

    def sparse_action_process(self, action):
        sparse_action = np.zeros((self.batch_size, self.max_task_size, self.max_par_size))
        for i in range(self.batch_size):
            for j in range(self.max_task_size):
                for k in range(self.max_par_size):
                    sparse_action[i][j][action[i][j] - 1] = 1
        sparse_action = np.reshape(sparse_action, [-1, self.max_par_size])
        return sparse_action

    def initial_lstm_state(self, batch_size = 1):
        self.lstm_state_h = np.random.normal(0,1,(batch_size, self.max_par_size))
        self.lstm_state_c = np.random.normal(0,1,(batch_size, self.max_par_size))
        self.lstm_state = [self.lstm_state_h, self.lstm_state_c]


    def train_model(self, data, action, reward, next_obs, epoch, direct = False):
        par_fea, task_fea, task_all = self.feature_extract(data, next_obs)
        self.initial_lstm_state(self.batch_size)
        print reward
        sparse_action = self.sparse_action_process(action)
        print "sparse_action"
        print sparse_action
        feed = {self.tf_par : par_fea, self.tf_task : task_fea, self.task_all : task_all, self.initial_state_c: self.lstm_state[0], self.initial_state_h: self.lstm_state[1], self.tf_action : sparse_action, self.tf_epr: reward, self.keep_prob : 0.5, self.is_training : True}
        _, loss_val = self.sess.run([self.train_op, self.loss], feed)
        if epoch % 10 == 0:
            print "epoch: {} loss: {}".format(epoch, loss_val)#, q, q_sum)


    def _optimize(self, multi_task = True):
        self.logit = self._forward(self.tf_par, self.tf_task)
        pre_loss = 0.0
        if multi_task:
            pre_loss = self._predict_task(self.task_all)
        self.loss = self._loss_f(self.loss_name) + pre_loss * 0.01
        self.prob = tf.reshape(self.logit, (-1, self.max_task_size, self.max_par_size))
        self.p = tf.nn.softmax(self.prob)
        #self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay, momentum=0.95, epsilon=0.01).minimize(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op =  tf.train.AdamOptimizer(self.learning_rate, epsilon = 0.000015).minimize(self.loss)
        self.train_op = tf.group([self.train_op, update_ops])
