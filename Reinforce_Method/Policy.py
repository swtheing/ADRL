import sys
import tensorflow as tf
import numpy as np
class Policy():
    def __init__(self, model, episilon):
        self.model = model
        self.episilon = episilon

    def expand_episilion(self, dis_epi):
        self.episilon -= dis_epi

    def action_sel(self, Observe):
        prob, Q = self.model.test_model(Observe)
        #print Q
        max_act_id = np.argmax(prob)
        if np.random.rand(1)[0] >= self.episilon:
            #return 3, Q[0, max_act_id], Q
            return max_act_id + 1, Q[0, max_act_id], Q
        act_id = np.random.randint(0, high=prob.shape[1])
        #return 3, Q[0, max_act_id], Q
        return act_id + 1, Q[0, max_act_id], Q

