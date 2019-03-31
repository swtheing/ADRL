import sys
import tensorflow as tf
import numpy as np
class Policy():
    def __init__(self, model, episilon):
        self.model = model
        self.episilon = episilon

    def expand_episilion(self, dis_epi, min_epi):
        if self.episilon > min_epi:
            self.episilon -= dis_epi

    def action_sel(self, Observe, max_sel = True, continues = False, multi_act = False):
        if not continues:
            if multi_act:
                if max_sel:
                    return None
                else:
                    prob, Q = self.model.test_model([Observe])
                    act_list = []
                    print prob[0]
                    for i in range(len(prob[0])):
                        act_list.append(np.random.choice(range(prob[0][i].shape[0]), p= prob[0][i]) + 1)
                    print "act_list:"
                    print act_list
                    return act_list, None, None
            else:
                prob, Q = self.model.test_model([Observe])
                max_act_id = np.argmax(Q)
                if max_sel:
                    if np.random.rand(1)[0] >= self.episilon:
                        return max_act_id + 1, Q[0, max_act_id], Q
                    act_id = np.random.randint(0, high=prob.shape[1])
                else:
                    act_id = np.random.choice(range(len(prob[0])), p= prob[0])
                return act_id + 1, Q[0, max_act_id], Q
        else:
            prob, Q = self.model.test_model(Observe)  
            return prob[0], Q, Q

    def copy_policy(self):
        self.model.copy()
