#!/usr/bin/python
# -*- coding: utf8 -*-

from data_util import env
#import gym
import numpy as np
import time
import os
import codecs
import math
import copy
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
        self.data_loader.load_task_static() # 读取yellow 数据
        self.data_loader.get_trajectories() # 读取uber数据
        #self.data_loader.overall_position_normalization()
        self.data_loader.get_merge_task(self.config.aim_day_num) 
        print "task generation"
        self.task_generator = TaskGenerator()
        self.task_generator.gen_task_list(self.data_loader.zip_data) # 按分布采样生成task列表
        self.task_generator.set_poisson_distribution(self.config.poisson_lamda, self.config.poisson_episode_num)
        print "trajectory sampling size: %d" % len(self.data_loader.trajectory_data)
        # new simulator
        self.simulator = StateSimulator()
        # reset
        self.simulator.trajector.init_sampling( \
            self.data_loader.trajectory_data, self.config.trajector_sampling_size)  # 采样生成路线数据
        self.episode_task_num = self.config.episode_task_num
        self.speed_init()
        #self.task_sampling()
        #self.preprocess()
        # clear memeory
        # self.self.data_loader.reset() 
        # clear log
        self.log_file = codecs.open(self.config.log_file_path, "w", "utf8")
        self.log_file.close()

    def reset(self, is_test=False): # 重新采样
        self.inner_step = 0
        self.simulator.trajector.init_sampling( \
            self.data_loader.trajectory_data, self.config.trajector_sampling_size)  # 采样生成路线数据
        self.task_sampling()
        self.speed_tune()
        self.preprocess()
        self.simulator.output_state(self.config.log_file_path, self.inner_step)
        print "reset done"
        if is_test:
            return self.simulator
        else:
            return self.simulator.new_feature

    def preprocess(self):
        self.simulator.clear()
        self.simulator.init_participants(self.config.participant_num) # 初始化n个taxi
        #print "step:0, task_num:%s" % len(task_samples)
        self.simulator.update_state(self.task_samples, [])
        self.simulator.new_feature = self.pre_process_feature()

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
        #self.simulator.trajector.set_ave_speed(self.config.default_ave_speed)
        print "ave speed: %s" % self.simulator.trajector.ave_speed

    def task_sampling(self):
        if self.config.env_var:
            self.task_generator.set_poisson_distribution(self.config.poisson_lamda, self.config.poisson_episode_num)
            self.task_samples = self.task_generator.task_sampling_poisson( \
                    self.inner_step, self.episode_task_num, self.config.max_task_size)
        else:
            self.task_samples = self.task_generator.task_sampling_default(self.episode_task_num)
            #self.task_samples = self.task_generator.task_sampling_random(self.episode_task_num)
        #self.task_samples = self.task_generator.task_sampling_default(self.episode_task_num)
        print "task sampling... new task num: %d" % len(self.task_samples)

    def render(self):
        return self.env.render()

    def step_raw(self, actions):
        self.inner_step += 1
        #actions = [action]
        self.speed_tune()
        self.task_sampling()
        self.simulator.update_state(self.task_samples, actions)
        #self.simulator.output_state(self.config.log_file_path, self.inner_step)
        done = False
        if self.inner_step == self.config.max_step:
            done = True
        info = None
        return self.simulator, self.simulator.reward, done, info

    def step(self, actions_pid_list, is_test=False):
        # init
        self.inner_step += 1
        print "INNER SETP:%s" % self.inner_step
        reward = 0
        done = False
        #actions = [action]
        print "ACTION:",
        print actions_pid_list
        self.speed_tune()
        self.task_sampling()
        actions = self.pre_process_action(actions_pid_list)
        self.simulator.update_state(self.task_samples, actions)
        
        ### rewarding
        #print self.simulator.reward
        dup_simulator = copy.deepcopy(self.simulator)
        is_finished = dup_simulator.update_state([], [])
        while_counter = 0
        while (not is_finished):
            while_counter += 1
            if while_counter > 1000:
                print "rewarding while overflow"
                break
            is_finished = dup_simulator.update_state([], [])
        #self.simulator.output_state(self.config.log_file_path, self.inner_step)

        # get feature
        self.simulator.new_feature = self.pre_process_feature()
        
        # dup reward
        self.simulator.final_reward = dup_simulator.reward

        # return result
        info = [0.0, 0.0, 0.0, 0.0, 0.0]
        if self.inner_step == self.config.max_step:
            done = True
            reward = self.simulator.final_reward
        else:
            reward = 0

        # DONE
        if done:
            # self.simulator.pending_time = dup_simulator.pending_time
            # self.simulator.total_fare_amount = dup_simulator.total_fare_amount
            self.simulator.task_pending_time = copy.deepcopy(dup_simulator.task_pending_time)
            self.simulator.participant_fare = copy.deepcopy(dup_simulator.participant_fare)
            self.simulator.finished_task_num = dup_simulator.finished_task_num
            # if self.simulator.finished_task_num > 0:
            #     ave_pending_time = self.simulator.pending_time/self.simulator.finished_task_num
            # else:
            #     ave_pending_time = 0.0
            # ave_fare_amount = self.simulator.total_fare_amount/len(self.simulator.participants)
            # info = [ave_pending_time, ave_fare_amount]

            # print "EPISODE_REWARD:%s\tPENDING_TIME:%s\tFARE_AMOUNT:%s\tTOTAL_PENDING_TIME:%s\tFINISHED_TASK:%s\tTOTAL_FARE:%s\tPAR_NUM:%s" \
            #     % (reward, ave_pending_time, ave_fare_amount,\
            #         self.simulator.pending_time, self.simulator.finished_task_num, \
            #         self.simulator.total_fare_amount, len(self.simulator.participants))

            task_time_cost = []
            par_fare_amount = []
            for tid in self.simulator.task_pending_time:
                #print "TASK %s,%s" % (tid, self.simulator.task_pending_time[tid])
                task_time_cost.append(self.simulator.task_pending_time[tid])
            for pid in self.simulator.participant_fare:
                #print "PAR %s,%s" % (pid, self.simulator.participant_fare[pid])
                par_fare_amount.append(self.simulator.participant_fare[pid])
            mean_time_cost = np.mean(task_time_cost)
            std_time_cost = np.std(task_time_cost)
            mean_fare_amount = np.mean(par_fare_amount)
            std_fare_amount = np.std(par_fare_amount)
            info = [mean_time_cost, std_time_cost, mean_fare_amount, std_fare_amount, self.simulator.finished_task_num]
            print "EPISODE_REWARD:%s\tFINISHED_NUM:%s\tTIME_MEAN:%s\tTIME_STD:%s\tFARE_MEAN:%s\tFARE_STD:%s" \
                % (reward, self.simulator.finished_task_num, mean_time_cost, \
                    std_time_cost, mean_fare_amount, std_fare_amount)

        del dup_simulator
        if is_test:
            return self.simulator, reward, done, info
        else:
            return self.simulator.new_feature, reward, done, info

    def pre_process_action(self, actions_pid_list):
        #[pid1,pid2,pid3,0,0,0,0,0,0,0]
        actions = []
        for index in range(len(self.simulator.new_task_list)):
            action_pid = actions_pid_list[index]
            if action_pid <= 0:
                continue
            action_taskid = self.simulator.new_task_list[index][0]
            action = ["pick", action_pid, action_taskid]
            actions.append(action)
        return actions

    def pre_process_feature(self):
        tmp_par_feature = np.zeros((self.config.max_par_size, self.config.par_feature_size))
        tmp_task_feature = np.zeros((self.config.max_task_size, self.config.task_feature_size))
        par_feature = np.zeros((self.config.max_par_size, 15))
        task_feature = np.zeros((self.config.max_task_size, 15))

        assert len(self.simulator.participants) <= self.config.max_par_size
        assert len(self.simulator.new_task_list) <= self.config.max_task_size

        for index in range(len(self.simulator.participants)):
            feature_list = []
            item_id = 0
            for item in self.simulator.participants[index+1]: # pid==index
                if item_id == 0:
                    item_id += 1
                    continue
                if isinstance(item, list):
                    for it in item:
                        feature_list.append(it)
                else:
                    feature_list.append(item)
            for i in range(len(feature_list)):
                feat = feature_list[i]
                if isinstance(feat, Enum):
                    feat = int(feat)
                par_feature[index][i] = feat
                item_id += 1
            # print index
            # print len(par_feature[index])
            # print par_feature[index]

        for index in range(len(self.simulator.new_task_list)):
            feature_list = []
            item_id = 0
            for item in self.simulator.new_task_list[index]: # pid==index
                if item_id == 0:
                    item_id += 1
                    continue
                if isinstance(item, list):
                    for it in item:
                        feature_list.append(it)
                else:
                    feature_list.append(item)
            for i in range(len(feature_list)):
                feat = feature_list[i]
                if isinstance(feat, Enum):
                    feat = int(feat)
                task_feature[index][i] = feat
                item_id += 1
            # print index
            # print len(task_feature[index])
            # print task_feature[index]
        # tmp_par_feature = par_feature[:,2:4]
        # tmp_task_feature = task_feature[:,2:4]
        # print tmp_par_feature
        feature = [par_feature, task_feature]
        return feature

