#!/usr/bin/python
# -*- coding: utf8 -*-

from data_util import env
#import gym
import numpy as np
import time
import os
import codecs
import math
#from env_trj import *
from env_trj.data_load import *
from env_trj.task_generator import *
from env_trj.state_generator import *

class env_trajectory(env):
    def __init__(self, name, config):
        env.__init__(self, name)
        #self.env = gym.make(name)
        self.inner_step = 0
        self.config = config
        # load data
        self.data_loader = DataLoader(
            self.config.task_data_path,self.config.trajectory_data_path)
        self.data_loader.get_merge_task(self.config.aim_day_num) # 读取yellow 数据
        print "task generation"
        self.task_generator = TaskGenerator()
        self.task_generator.gen_task_list(self.data_loader.zip_data) # 按分布采样生成task列表
        self.task_generator.set_poisson_distribution(self.config.poisson_lamda, self.config.poisson_episode_num)
        print "trajectory sampling"
        self.data_loader.get_trajectories() # 读取uber数据
        print "trajectory size: %d" % len(self.data_loader.trajectory_data)
        # new simulator
        self.simulator = StateSimulator()
        # reset
        self.simulator.trajector.init_sampling( \
            self.data_loader.trajectory_data, self.config.trajector_sampling_size)  # 采样生成路线数据
        self.episode_task_num = self.config.episode_task_num
        self.speed_init()
        self.task_sampling()
        self.preprocess()
        # clear memeory
        # self.self.data_loader.reset() 
        # clear log
        self.log_file = codecs.open(self.config.log_file_path, "w", "utf8")
        self.log_file.close()

    def reset(self): # 重新采样
        self.inner_step = 0
        self.simulator.trajector.init_sampling( \
            self.data_loader.trajectory_data, self.config.trajector_sampling_size)  # 采样生成路线数据
        self.speed_tune()
        self.task_sampling()
        self.preprocess()
        self.simulator.output_state(self.config.log_file_path, self.inner_step)
        print "reset done"
        return self.simulator

    def speed_init(self):
        if self.config.env_var:
            self.simulator.trajector.speed_tuner_init( \
                self.config.normal_mu, self.config.normal_sigma, self.config.normal_episode_num)
        self.speed_tune()

    def speed_tune(self):
        if self.config.env_var:
            self.simulator.trajector.speed_tuner(self.config.default_ave_speed, self.inner_step)
        else:
            self.simulator.trajector.set_ave_speed(self.config.default_ave_speed)
        print "ave speed: %s" % self.simulator.trajector.ave_speed

    def task_sampling(self):
        if self.config.env_var:
            self.task_samples = self.task_generator.task_sampling_poisson(self.inner_step, self.episode_task_num)
        else:
            self.task_samples = self.task_generator.task_sampling_random(self.episode_task_num)
        print "task sampling... new task num: %d" % len(self.task_samples)

    def render(self):
        return self.env.render()

    def preprocess(self):
        self.simulator.task_key_gen = -1
        self.simulator.running_schedules = dict()
        self.simulator.pending_schedules = dict()
        self.simulator.finished_schedules = dict()
        self.simulator.participants = dict()
        self.simulator.reward = 0.0
        self.simulator.init_participants(self.config.participant_num) # 初始化n个taxi
        #print "step:0, task_num:%s" % len(task_samples)
        self.simulator.update_state(self.task_samples, [])

    def step(self, actions):
        self.inner_step += 1
        #actions = [action]
        self.speed_tune()
        self.simulator.update_state(self.task_samples, actions)
        self.simulator.output_state(self.config.log_file_path, self.inner_step)
        self.task_sampling()
        done = False
        if self.inner_step == self.config.max_step:
            done = True
        info = None
        return self.simulator, self.simulator.reward, done, info


