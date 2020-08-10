import sys
import os
import numpy as np
from data_util import *
import linecache
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def string(list):
    info = "["
    for i in list[:-1]:
        info += str(i) + ","
    info += str(list[-1]) + "]"
    return info

def set_lr(action, filename):
    file = open(filename, "r")
    file_data = ""
    for line in file:
        ans = ""
        if "lr_prob =" in line:
            ans = "    lr_prob = " + string(action) + "\n"
        elif "sample = " in line:
            #if action == "Start":
            #    ans = "    sample = 'Sample'\n"
            #else:
            ans = "    sample = 'Prob'\n"
        else:
            ans = line
        file_data += ans
    with open(filename,"w") as f:
        f.write(file_data)

def read_state(file):
    lr_list = []
    for line in open(file):
        last_line = line
    groups = last_line.strip().split("\t")
    for i in range(1, len(groups)):
        lr_list.append(float(groups[i]))
    return lr_list[0:-1], lr_list[-1]

def read_restore(file):
    for line in open(file):
        last_line = line
    groups = last_line.strip().split("\t")
    return groups[1], groups[2]

class env_lm(env):
    def __init__(self, name, config):
        env.__init__(self, name)
        self.count = 0
        self.max_step = config.max_step
        self.early_stop = config.early_stop
        self.outfile = config.outfile
        self.cmd = config.cmd
        self.lm_conf = config.lm_conf
        self.restore = config.restore
        #set_lr([-100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0], self.lm_conf)
    
    def scale_obs(self, obs):
        return np.divide(obs, 1000.0); 
        #return (obs - np.mean(obs)) / np.std(obs)

    def reset(self):
        self.count = 0
        set_lr([-100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0], self.lm_conf)
        cmd = self.cmd + " sss 0"
        os.system(cmd)
        #fp.read()
        obs, info = read_state(self.outfile)
        self.count += 1
        self.last_info = info
        self.min_info = info
        return self.scale_obs(obs)
    
    def step(self, action):
        set_lr(action, self.lm_conf)
        ckpt_path, restore_epoch = read_restore(self.restore)
        cmd = self.cmd + " " + str(ckpt_path) + " " + str(restore_epoch) 
        fp = os.system(cmd)
        #fp.read()
        obs, info = read_state(self.outfile)
        self.count += 1
        if info < self.min_info:
            self.min_info = info
        reward = 0.0
        done = False
        if self.count == self.max_step or (info > self.last_info and self.early_stop):
            done = True
            reward = -self.min_info
        return self.scale_obs(obs), reward, done, info 
