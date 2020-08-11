import tensorflow as tf
import numpy as np
from model import *
from T2T import *
from T2T_BN import *
np.set_printoptions(threshold=np.inf) 
class Trans_Ptr(model):
    def __init__(self, game_name, path, config, loss_name, attribute = "reserve", copy_model = None, new_graph = False):
        self.max_par_size = config.max_par_size
        self.max_task_size = config.max_task_size
        self.par_feature_size = config.par_feature_size
        self.task_feature_size = config.task_feature_size
        self.num_block = config.num_block
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.max_par_size = config.max_par_size
        self.out_size = config.max_par_size
        self.action_size = self.out_size * config.max_task_size
        self.gamma = config.gamma
        self.decay = config.decay
        self.learning_rate = config.learning_rate
        self.loss_name = loss_name
        self.dropout_prob = config.dropout_prob
        self.batch_size = config.batch_size
        self.value_coffe = config.value_coffe
        self.entro_coffe = config.entro_coffe
        self.epp = config.epp
        self.attri = attribute
        self.initial_lstm_state()
        self.var_list = []
        self.multi_task = config.multi_task
        graph = tf.get_default_graph()
        with graph.as_default():
             self._build_model()
             init = [tf.global_variables_initializer(), tf.local_variables_initializer()] 
             model.__init__(self, "Trans_tr", game_name, config, copy_model = copy_model)
        if copy_model == None:
            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(graph=graph, config=session_config)
        else:
            self.sess = copy_model.sess
        if not new_graph:
            self.sess.run(init)
         




    def feature_extract(self, data, next_obs = None, rescale = True):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        task_fea = np.zeros([len(data), self.max_task_size, self.task_feature_size])
        par_fea = np.zeros([len(data), self.out_size, self.par_feature_size])
        task_all = np.zeros([len(data), self.max_task_size, self.task_feature_size])
        tf_rule_act = np.zeros([len(data), self.action_size])
        for i in range(len(data)):
            tf_rule_act[i,:] = self.cal_rule_act(data[i])[0]
            par_fea[i,:,:] = data[i][0]
            task_fea[i,:,:] = data[i][1]
            task_all[i,:,:] = data[i][1]
            if next_obs != None:
                count = 0
                for j in range(task_all.shape[1]):
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
        return par_fea, task_fea, task_all, tf_rule_act

    def _build_model(self):
        self._create_placeholders()
        self._optimize(self.multi_task)
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.attri)
    
    def _create_placeholders(self):
        self.tf_par = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size, self.par_feature_size], name="tf_par")
        self.tf_task = tf.placeholder(dtype=tf.float32, shape=[None, self.max_task_size, self.task_feature_size], name="tf_task")
        self.tf_rule_action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name="tf_rule_action")
        self.tf_action = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size], name="tf_action")
        self.initial_state_c = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size], name="tf_state_c")
        self.initial_state_h = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size], name="tf_state_h")
        self.tf_old_ratios = tf.placeholder(dtype=tf.float32, shape=[None, self.max_task_size], name="tf_old_ratios") 
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")
        self.task_all = tf.placeholder(dtype=tf.float32, shape=[None, self.max_task_size, self.task_feature_size], name="task_all")
        self.tf_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_q")
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="tf_keep_prob")

    def _predict_task(self, tf_predict):
        self.task_predict = tf.layers.dense(self.dec, self.task_feature_size)
        mse = tf.square(tf_predict - self.task_predict)
        return tf.reduce_mean(mse)
    
    def _mask(self, inputs, outputs):
        #inputs = tf.Print(inputs, [inputs], summarize = 80, message = "mask_input")
        dims = []
        for i in range(len(inputs.get_shape())):
            dims.append(1)
        K = outputs.get_shape().as_list()[-1]
        dims[-1] = K
        outputs_zeros = tf.zeros_like(outputs)
        outputs_mask = tf.tile(tf.reduce_sum(inputs, keep_dims = True, axis = -1), dims)
        outputs = tf.where(tf.equal(outputs_mask, 0), outputs_zeros, outputs)
        #:utputs = tf.Print(outputs, [outputs], summarize = 80, message = "mask_output")
        return outputs
    
    def _forward(self, tf_par, tf_task):  # x ~ [1,D]
        #tf_task = tf.Print(tf_task, [tf_task], summarize = 80, message = "tf_task")
        #tf_par = tf.Print(tf_par, [tf_par], summarize = 80, message = "tf_par")
        tf_par_norm = tf.layers.batch_normalization(tf_par, momentum = 0.1, training = self.is_training)
        #tf_par_norm = tf.Print(tf_par_norm, [tf_par_norm], summarize = 80, message = "tf_par_norm")
        tf_task_norm = tf.layers.batch_normalization(tf_task, momentum = 0.1, training = self.is_training)
        tf_task_norm = self._mask(tf_task, tf_task_norm)
        #tf_task_norm = tf.Print(tf_task_norm, [tf_task_norm], summarize = 20, message = "tf_task")
        par_emb = tf.layers.dense(tf_par_norm, self.hidden_size, tf.nn.relu)
        #par_emb = tf.Print(par_emb, [par_emb], summarize = 80, message = "par_emb")
        task_emb_norm = tf.layers.dense(tf_task_norm, self.hidden_size, tf.nn.relu)
        task_emb = self._mask(tf_task_norm, task_emb_norm)
        #task_emb = tf.Print(task_emb, [task_emb], summarize = 80, message = "task_emb")
        #task_emb = tf.keras.layers.LSTM(self.hidden_size, return_sequences = True)(task_emb)
        #self.T2T = T2T_Model(None, self.hidden_size, self.dropout_prob, self.num_block, self.num_heads, self.hidden_size, self.is_training)
        self.T2T = T2T_BN_Model(None, self.hidden_size, self.dropout_prob, self.num_block, self.num_heads, self.hidden_size, self.is_training)
        self.dec, pointer = self.T2T.Trans_Pointer(par_emb, task_emb, emb = False)
        # pointer = tf.Print(pointer, [pointer], message = "pointer", summarize = 20)
        # print pointer.shape.as_list()
        print self.initial_state_c.shape.as_list()
        print self.initial_state_h.shape.as_list()
        #pointer = tf.keras.layers.LSTM(self.out_size, return_sequences = True)(pointer)
        #print pointer.shape.as_list()
        #self.state = [lstm_c, lstm_h]
        logit = tf.reshape(pointer, shape = (-1, self.out_size))
        logit += tf.layers.dense(tf.reshape(self.tf_rule_action, shape = (-1, self.out_size)), self.out_size)
        v_predict = tf.layers.dense(self.dec, 1)[0,0,:]
        return logit, v_predict

    def _loss_f(self, loss_name):
        ##No MSE 
        if loss_name == "MSE":
            mse = tf.square(self.tf_Q - self.v_predict)
            return mse
        elif loss_name == "CE":
            self.logit = tf.reshape(self.logit, shape = (-1, self.out_size))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.tf_action)
            cross_entropy = tf.reshape(cross_entropy, shape = [-1, self.max_task_size])
            #cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_1")
            mask_task = tf.reduce_sum(self.tf_task, axis = -1)
            paddings = tf.zeros_like(cross_entropy)
            cross_entropy = tf.where(tf.equal(mask_task, 0), paddings, cross_entropy)
            #cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_2")
            cross_entropy = tf.reduce_sum(cross_entropy, axis = -1)
            #cross_entropy = tf.Print(cross_entropy, [cross_entropy], summarize = 20, message = "cross_entropy_3")
            return tf.reduce_mean(cross_entropy * self.tf_epr)
        elif loss_name == "CLP":
            #self.q = tf.multiply(self.logit, self.tf_action)
            #self.q_sum = tf.reduce_sum(self.q, axis=1)
            self.delta = self.tf_Q - self.v_predict
            self.clipped_error = tf.where(tf.abs(self.delta) < 0.3,
                                    0.5 * tf.square(self.delta),
                                    tf.abs(self.delta) - 0.5)
            return tf.reduce_sum(self.clipped_error)
        elif loss_name == "PPO_L":
            act_oht = tf.reshape(self.tf_action, [-1, self.max_task_size, self.out_size])
            self.tf_new_ratios = tf.reduce_sum(self.p * act_oht, axis = -1)
            self.tf_ratios = self.tf_new_ratios / self.tf_old_ratios
            clipped_ratios = tf.clip_by_value(self.tf_ratios, clip_value_min= 1 - self.epp, clip_value_max=1 + self.epp)
            clipped_ratios = tf.Print(clipped_ratios, [clipped_ratios], message = "clip_ratios", summarize = 80)
            self.adv = tf.tile(tf.reshape(self.tf_epr, [-1, 1]), [1, self.max_task_size])
            loss_clip = tf.minimum(tf.multiply(self.tf_ratios, self.adv), tf.multiply(clipped_ratios, self.adv))
            loss_clip = tf.Print(loss_clip, [loss_clip], message = "loss_clip", summarize = 80)
            return -tf.reduce_mean(loss_clip, axis = -1)
    
    def distance(self, gps_1, gps_2):
        return pow((pow((gps_1[0] - gps_2[0]), 2) + pow((gps_1[1] - gps_2[1]), 2)), 0.5)

    def cal_rule_act(self, obs):
        par_f = obs[0]
        task_f = obs[1]
        tf_rule_action = np.zeros((self.max_task_size, self.max_par_size))
        unavailable_pid_list = []
        for index in range(len(par_f)):
            participant = par_f[index]
            pid = index
            p_state = int(participant[0])
            if int(p_state) != 1:
                unavailable_pid_list.append(pid)
        for index in range(len(task_f)):
            task = task_f[index]
            t_start_pos = [task[0], task[1]]
            min_id = -1
            min_dist =10000000.0
            for pid in range(len(par_f)):
                p_start_pos = [par_f[pid-1][3], par_f[pid-1][4]]
                dist = self.distance(t_start_pos, p_start_pos)
                if pid not in unavailable_pid_list:
                    tf_rule_action[index, pid] = 1.0 / dist
                    if dist < min_dist:
                        min_dist = dist
                        min_id = pid
                unavailable_pid_list.append(min_id)
        return np.reshape(tf_rule_action, [1, -1])
    
    def test_model(self, data):
        par_fea, task_fea, task_all, tf_rule_act = self.feature_extract(data)
        self.initial_lstm_state(1)
        #print "feature:"
        #print task_fea
        #print par_fea
        feed = {self.tf_par : par_fea, self.tf_task : task_fea, self.tf_rule_action : tf_rule_act, self.initial_state_c: self.lstm_state[0], self.initial_state_h: self.lstm_state[1], self.keep_prob: 1.0, self.is_training: False}
        p, Q = self.sess.run([self.p, self.v_predict], feed)
        print self.lstm_state[0].shape
        return p, Q[0]

    def sparse_action_process(self, action):
        print "p_action"
        #print action
        sparse_action = np.zeros((self.batch_size, self.max_task_size, self.out_size))
        for i in range(self.batch_size):
            for j in range(self.max_task_size):
                for k in range(self.out_size):
                        sparse_action[i][j][action[i][j] - 1] = 1
        sparse_action = np.reshape(sparse_action, [-1, self.out_size])
        return sparse_action

    def initial_lstm_state(self, batch_size = 1):
        self.lstm_state_h = np.random.normal(0,1,(batch_size, self.out_size))
        self.lstm_state_c = np.random.normal(0,1,(batch_size, self.out_size))
        self.lstm_state = [self.lstm_state_h, self.lstm_state_c]


    def train_model(self, data, action, reward, value, next_obs, rule_act, epoch, ratios = None, direct = False):
        par_fea, task_fea, task_all, tf_rule_act = self.feature_extract(data, next_obs)
        self.initial_lstm_state(self.batch_size)
        print reward
        sparse_action = self.sparse_action_process(action)
        print "sparse_action"
        feed = {self.tf_par : par_fea, self.tf_task : task_fea, self.tf_rule_action : tf_rule_act, self.task_all : task_all, self.initial_state_c: self.lstm_state[0], self.initial_state_h: self.lstm_state[1], self.tf_action : sparse_action, self.tf_epr: reward, self.keep_prob : 0.5, self.is_training : True, self.tf_Q : value, self.tf_old_ratios : ratios}
        _, loss_val = self.sess.run([self.train_op, self.loss], feed)
        if epoch % 10 == 0:
            print "epoch: {} loss_name {} loss: {}".format(epoch, self.loss_name, loss_val)#, q, q_sum)


    def _optimize(self, multi_task = True):
        with tf.variable_scope(self.attri):
            self.logit, self.v_predict = self._forward(self.tf_par, self.tf_task)
        pre_loss = 0.0
        self.prob = tf.reshape(self.logit, (-1, self.max_task_size, self.out_size))
        self.p = tf.nn.softmax(self.prob)
        if multi_task:
            pre_loss = self._predict_task(self.task_all)
        self.loss = self._loss_f(self.loss_name) + pre_loss * 0.01 + self._loss_f("MSE") * self.value_coffe
        self.loss = tf.Print(self.loss, [self.loss], message = "loss", summarize = 80)
        #self.loss = self._loss_f("CLP")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op =  tf.train.AdamOptimizer(self.learning_rate, epsilon = 0.000015).minimize(self.loss)
        self.train_op = tf.group([self.train_op, update_ops])
  
