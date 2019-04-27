import sys
import os
import tensorflow as tf
class model():
    def __init__(self, model_name, game_name, config, copy_model = None):
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.model_name = model_name
        self.game_name = game_name
        if copy_model != None:
            self.assign(copy_model.get_params())

    def feature_extract(self, data):
        raise NotImplementedError("Abstract Method")
    
    def train_model(self, data, action, reward, value, epoch, direct = False):
        raise NotImplementedError("Abstract Method")
    
    def test_model(self, data):
        raise NotImplementedError("Abstract Method")
    
    def get_params(self):
        return self.var_list

    def assign(self, value):
        copy = []
        for i in range(len(self.var_list)):
            copy.append(tf.assign(self.var_list[i], value[i]))
        self.copy_op = tf.group(*copy, name='copy_op')
    
    def copy(self):
        copy_op = self.sess.run(self.copy_op)

    def restore_model(self, save_path):
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            print "no saved model to load. starting new session"
            return 0
        else:
            print "loaded model: {}".format(load_path)
            epoch = int(load_path.split('-')[-1])
            return epoch + 1

    def save_model(self, save_path, epoch):
        save_floder = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_floder):
            os.makedirs(save_floder)
        self.saver.save(self.sess, save_path, global_step = epoch)
        print "#{} SAVED MODEL #{}".format(self.model_name, epoch)



