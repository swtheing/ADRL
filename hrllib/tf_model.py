from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils import try_import_tf
import tensorflow as tf
import numpy as np

class Dense(TFModelV2):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.var_list = []
        self.feature_size = 16
        self._create_w();
        self.register_variables(self.var_list)
    
    def _create_w(self):
        with tf.name_scope("data"):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(self.num_outputs), dtype=tf.float32)
            self.W1 = tf.get_variable("W1", [self.feature_size, self.num_outputs], initializer=xavier_l1)
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(1.0), dtype=tf.float32)
            self.W2 = tf.get_variable("W2", [self.feature_size, 1.0], initializer=xavier_l2)
        self.var_list.append(self.W1)
        self.var_list.append(self.W2) 
    
    def forward(self, input_dict, state, seq_lens):
        model_out = tf.matmul(input_dict['obs'], self.W1)
        self._value_out = tf.matmul(input_dict['obs'], self.W2)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


