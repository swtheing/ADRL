#!/usr/bin/python
# -*- coding: utf8 -*-

from data_util import env
#import gym
import numpy as np
import time
import os
import codecs
#from env_trj import *
from env_trj.data_load import *
from env_trj.task_generator import *
from env_trj.state_generator import *

class env_trajectory(env):
    def __init__(self, name, config):
        env.__init__(self, name)
        self.config = config
        self.step_num = 0
        self.data_loader = DataLoader(
            config.task_data_path,config.trajectory_data_path)
        self.data_loader.get_merge_task(config.aim_day_num) # 读取yellow 数据
        print "task generation"
        self.task_generator = TaskGenerator()
        self.task_generator.gen_task_list(self.data_loader.zip_data) # 按分布采样生成task列表
        print "trajectory sampling"
        self.data_loader.get_trajectories() # 读取uber数据
        print "trajectory size: %d" % len(self.data_loader.trajectory_data)
        # new simulator
        self.simulator = StateSimulator()
        # reset
        self.simulator.trajector.init_sampling(self.data_loader.trajectory_data, config.trajector_sampling_size)  # 采样生成路线数据
        self.ave_speed = self.simulator.trajector.set_ave_speed(config.default_ave_speed)
        print "ave speed: %s" % self.ave_speed
        print "task sampling"
        self.episode_task_num = config.episode_task_num
        self.task_samples = self.task_generator.task_generation(self.episode_task_num)
        self.preprocess(config)
        # clear memeory
        # self.self.data_loader.reset() 
        # clear log
        self.log_file = codecs.open(config.log_file_path, "w", "utf8")
        self.log_file.close()
        self.simulator.output_state(config.log_file_path, 0)

    def load(self, config):
        self.data_loader = DataLoader(
            config.task_data_path,config.trajectory_data_path)
        self.data_loader.get_merge_task(config.aim_day_num) # 读取yellow 数据
        print "task generation"
        self.task_generator = TaskGenerator()
        self.task_generator.gen_task_list(self.data_loader.zip_data) # 按分布采样生成task列表
        print "trajectory sampling"
        self.data_loader.get_trajectories() # 读取uber数据
        print "trajectory size: %d" % len(self.data_loader.trajectory_data)

    def reset(self): # 重新采样
        config = self.config
        self.step_num = 0
        self.simulator.trajector.init_sampling(self.data_loader.trajectory_data, config.trajector_sampling_size)  # 采样生成路线数据
        self.ave_speed = self.simulator.trajector.set_ave_speed(config.default_ave_speed)
        print "ave speed: %s" % self.ave_speed
        print "task sampling"
        self.episode_task_num = config.episode_task_num
        self.task_samples = self.task_generator.task_generation(self.episode_task_num)
        self.preprocess(config)
        self.simulator.output_state(config.log_file_path, 0)
        print "reset done"
        return self.simulator

    def task_sampling(self):
        self.task_samples = self.task_generator.task_generation(self.episode_task_num)

    def render(self):
        return self.env.render()

    def preprocess(self, config):
        self.simulator.task_key_gen = 0
        self.simulator.running_schedules = dict()
        self.simulator.pending_schedules = dict()
        self.simulator.finished_schedules = dict()
        self.simulator.participants = dict()
        self.simulator.reward = 0.0
        self.simulator.init_participants(config.participant_num) # 初始化n个taxi
        #print "step:0, task_num:%s" % len(task_samples)
        self.simulator.update_state(self.task_samples, [])

    def step(self, action):
        config = self.config
        self.step_num += 1
        actions = [action]
        self.simulator.update_state(self.task_samples, actions)
        self.simulator.output_state(config.log_file_path, self.step_num)
        self.task_sampling()
        done = False
        if self.step_num == config.max_step:
            done = True
        info = None
        return self.simulator, self.simulator.reward, done, info

