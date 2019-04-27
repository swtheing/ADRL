#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs
from enum import Enum
from enum import IntEnum
import random
import math
from math import radians, cos, sin, asin, sqrt
import numpy as np

class TaskState(IntEnum):
    pending = 0
    picking = 1
    picked = 2
    finished = 3


class ParticipantState(IntEnum):
    available = 0
    working = 1
    unavailable = -1


class Trajectory():
    def __init__(self):
        self.trajectories = []
        self.ave_speed = 0.0
        self.speed_normal_distribution = []

    def init_sampling(self, trajectory_data, sampling_size=10000):
        """
        input raw trajectory_data and sampling trajectories
        """
        #raw_trajectories = random.sample(trajectory_data, sampling_size)
        raw_trajectories = trajectory_data[:sampling_size] # non random
        self.trajectories = [x[2] for x in raw_trajectories]

    def get_distance(self, x1, y1, x2, y2):
        return pow((pow((x1 - x2), 2) + pow((y1 - y2), 2)), 0.5)

    def speed_tuner_init(self, mu, sigma, total_size):
        speed_normal_list = sorted(np.random.normal(mu, sigma, total_size))
        self.speed_normal_distribution = speed_normal_list
        for s in reversed(speed_normal_list):
            self.speed_normal_distribution.append(s)
        #print self.speed_normal_distribution

    def speed_tuner(self, ave_speed_base, step_num):
        speed_index = step_num % len(self.speed_normal_distribution)
        speed_tuner = self.speed_normal_distribution[speed_index]
        if speed_tuner < -0.99:
            speed_tuner = -0.99
        self.ave_speed = (1 + speed_tuner) * ave_speed_base
        return self.ave_speed

    def set_ave_speed(self, trajectory_data, sampling_size=1000):
        sum_distance = 0.0
        sum_duration = 0.0
        full_size = len(trajectory_data)
        sampling_index_list = []
        for j in range(sampling_size):
            i = random.randint(0, full_size - 2)
            if (trajectory_data[i][0] != trajectory_data[i + 1][0]): # participant id should be same
                i -= 1
                continue
            start_t = trajectory_data[i][1][6]
            end_t = trajectory_data[i + 1][1][6]
            duration = end_t - start_t
            sum_duration += duration

            start_x, start_y = trajectory_data[i][2]
            end_x, end_y = trajectory_data[i + 1][2]
            distance = self.get_distance(start_x, start_y, end_x, end_y)
            sum_distance += distance
        # print sum_duration
        # print sum_distance
        self.ave_speed = sum_distance/sum_duration
        return self.ave_speed

    def set_ave_speed(self, ave_speed):
        self.ave_speed = ave_speed
        return self.ave_speed

    def yeild_rand_position(self):
        #random.shuffle(self.trajectories)
        index = random.randint(0, self.trajectories.size() - 1)
        return self.trajectories[index]

    def get_position(self, pid):
        return self.trajectories[pid]

    def haversine(self, lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2.0 * asin(sqrt(a)) 
        r = 6371.0 # 地球平均半径，单位为公里
        return c * r * 1000.0

    def move_simulate(self, start_pos, target_pos, speed, unit_time=300):
        x1, y1 = start_pos
        x2, y2 = target_pos
        x3, y3 = start_pos
        distance = self.get_distance(x1, y1, x2, y2)
        move_dis = speed * unit_time
        if move_dis > distance:
            # finish
            return target_pos, 0.0, distance
        if x2 != x1 and y2 != y1:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - (k * x1)
            theta = math.atan(abs(x1 - x2) / abs(y1 - y2))
            delta_y = 0.0 - move_dis * math.cos(theta)
            if y1 <= y1 - delta_y <= y2:
                y3 = y1 - delta_y
            else:
                y3 = y1 + delta_y
            x3 = (y3 - b) / k
        elif x2 != x1 and y2 == y1:
            y3 = y1
            if x2 > x1:
                x3 = x1 + move_dis
            else:
                x3 = x1 - move_dis
        else:
            delta_y = move_dis
            if y1 <= y1 - delta_y <= y2:
                y3 = y1 - delta_y
            else:
                y3 = y1 + delta_y
            x3 = x1
        new_pos = [x3, y3]
        return new_pos, (distance - move_dis), move_dis

class StateSimulator():
    def __init__(self):
        self.task_key_gen = 0
        self.running_schedules = dict()
        self.pending_schedules = dict()
        self.finished_schedules = dict()
        self.participants = dict()
        self.trajector = Trajectory()
        self.new_task_list = []
        self.reward = 0.0
        self.final_reward = 0.0
        self.pending_actions = []
        self.new_feature = []
        self.exe_actions = []

    def clear(self):
        self.task_key_gen = 0
        self.running_schedules = dict()
        self.pending_schedules = dict()
        self.finished_schedules = dict()
        self.participants = dict()
        self.reward = 0.0
        self.final_reward = 0.0
        self.new_task_list = []
        self.pending_actions = []
        self.new_feature = []
        self.exe_actions = []

    def init_participants(self, participants_num=10):
        for pid in range(1, participants_num + 1):
            participant_id = pid
            # start_pos = self.trajector.yeild_rand_position()
            # target_pos = self.trajector.yeild_rand_position()
            start_pos = self.trajector.get_position(pid)
            target_pos = self.trajector.get_position(pid)
            cur_pos = start_pos
            remain_distance = self.trajector.get_distance( \
                start_pos[0], start_pos[1], target_pos[0], target_pos[1])
            cur_cost_dis = 0.0
            task_id = 0
            speed = self.trajector.ave_speed
            state = ParticipantState["available"]
            participant = [participant_id, state, task_id, \
                cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed]
            self.participants[participant_id] = participant

    def free_working_participant(self, participant_id):
        participant_id, state, task_id, \
            cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed = self.participants[participant_id]
        # target_pos = self.trajector.yeild_rand_position()
        target_pos = self.trajector.get_position(participant_id)
        cur_pos = start_pos
        remain_distance = self.trajector.get_distance( \
            start_pos[0], start_pos[1], target_pos[0], target_pos[1])
        cur_cost_dis = 0.0
        task_id = 0
        state = ParticipantState["available"]
        self.participants[participant_id] = [participant_id, state, task_id, \
            cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed]

    def update_state(self, new_tasks, actions, unit_time=300):
        """
        2.update positon
        3.execute action
        1.update new tasks
        """
        is_finished = False
        self.exe_actions = []
        self.execute_action(actions)
        if len(self.running_schedules) == 0 \
                and len(self.pending_actions) == 0:
            is_finished = True
        self.update_position(unit_time)
        self.update_new_tasks(new_tasks)
        return is_finished

    def update_position(self, unit_time=300):
        for p_id in self.participants:
            p_id, state, task_id, \
                cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed = self.participants[p_id]
            if state != ParticipantState["unavailable"]:
                ave_speed = self.trajector.ave_speed
                self.participants[p_id][8] = ave_speed
                new_pos, remain_distance, cur_cost_dis = \
                    self.trajector.move_simulate(cur_pos, target_pos, ave_speed, unit_time)
                cur_pos = new_pos
                if remain_distance == 0: # finished, state change
                    if state == ParticipantState["working"]: #已接单
                        if task_id in self.running_schedules:
                            task = self.running_schedules[task_id]
                            state = task[1]
                            task[11] = cur_cost_dis
                            if state == TaskState["picking"]: #接到人, refresh task
                                state = TaskState["picked"]   #开始执行订单
                                start_pos = task[4]
                                target_pos = task[5]
                                task[3] = start_pos
                                task[1] = state
                                self.running_schedules.pop(task_id)
                                self.running_schedules[task_id] = task
                                # continue current traj
                                self.participants[p_id][3] = cur_pos # reset start pos
                                self.participants[p_id][6] = remain_distance
                                self.participants[p_id][7] = cur_cost_dis
                            elif state == TaskState["picked"]: #完成订单
                                self.finished_schedules[task_id] = task
                                self.finished_schedules[task_id][1] = TaskState["finished"]
                                self.running_schedules.pop(task_id)
                                self.free_working_participant(p_id)
                    else:
                        self.free_working_participant(p_id)
                else:
                    # continue random walk
                    self.participants[p_id][3] = cur_pos # reset start pos
                    self.participants[p_id][6] = remain_distance
                    self.participants[p_id][7] = cur_cost_dis

    def execute_action(self, new_actions):
        """
        先执行pending actions
        后执行new actions
        根据未执行的action生成pending actions
        """
        print "## exe actions:"
        tmp_actions = []
        for action in self.pending_actions:
            if not self.execution(action):
                tmp_actions.append(action)
        for action in new_actions:
            if not self.execution(action):
                tmp_actions.append(action)
        self.pending_actions = tmp_actions
        print "## pending:",
        print tmp_actions
        self.reward_compute()

    def execution(self, action):
        action_name, p_id, t_id = action
        print "##  %s" % action
        if p_id in self.participants \
                and t_id in self.pending_schedules \
                and self.participants[p_id][1] == ParticipantState["available"] \
                and self.pending_schedules[t_id][1] == TaskState["pending"]:
            self.exe_actions.append(action)
            print "##  %s" % self.participants[p_id][1],
            # update participant
            p_id, state, p_task_id, \
                cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed = self.participants[p_id]
            state = ParticipantState["working"]
            target_pos = self.pending_schedules[t_id][4]
            start_pos = cur_pos
            remain_distance = self.trajector.get_distance(cur_pos[0], cur_pos[1], target_pos[0], target_pos[1])
            self.participants[p_id] = [p_id, state, t_id, \
                cur_pos, start_pos, target_pos, remain_distance, cur_cost_dis, speed]
            print "##  %s" % self.participants[p_id][1]
            # update task
            self.running_schedules[t_id] = self.pending_schedules[t_id]
            self.running_schedules[t_id][1] = TaskState["picking"]  #update state
            self.running_schedules[t_id][2] = p_id                  #update participant_id
            self.pending_schedules.pop(t_id)
            return True #exe succ
        return False

    def reward_compute(self):
        # one step reward, no pending task
        # reward = finish_num + pos_distance * 3 - neg_distance - waiting_time
        print "## reward computing:",
        for task_id in self.finished_schedules:
            task = self.finished_schedules[task_id]
            if task[9] == 0:
                self.reward += task[10] * 5.0  #-1.0抵消成本，+3.0计算reward
                print "+%f\t" % (task[10] * 5.0),
                self.finished_schedules[task_id][9] = 1
        for pid in self.participants:
            state = self.participants[pid][1]
            cur_cost_dis = self.participants[pid]
            if state == ParticipantState["working"]:
                self.reward -= self.participants[pid][7]  #1.0抵消成本
                print "-%s\t" % self.participants[pid][7],
        for tid in self.pending_schedules:
            self.reward -= 0.01
            print "-%s\t" % "0.01",
        print "\n## reward: %s" % self.reward
        return self.reward

    def update_new_tasks(self, new_tasks):
        """
        args:  new_tasks:
        [[task_id,key_sec,
          time, [start_t, end_t] (year, month, day, hour, minute, second, unix_time, time_str)
          position, (start_x, end_x, start_y, end_y)
          passenger_num, distance, fare_amount]] 
        """
        self.new_task_list = []
        for task_data in new_tasks:
            self.task_key_gen += 1      #start from 1
            # task_id = "task-%d" % self.task_key_gen                  #id[0]
            task_id = self.task_key_gen
            state = TaskState["pending"]                            #state[1]
            participant_id = -1                                     #pid[2]
            cur_pos = task_data[3][0:2]                             #当前位置[3]
            start_pos = cur_pos                                     #起点[4]
            end_pos = task_data[3][2:4]                             #终点[5]
            recruiter_num, distance, fare_amount = task_data[4:7]   #乘客人数[6]，距离[7]，花费[8]
            reward_state = 0                                        #reward计算状态[9]
            real_dis = self.trajector.get_distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1]) # 真实需要移动距离[10]
            move_dis = 0.0                                          #当前step移动距离[11]
            task = [task_id, state, participant_id, cur_pos, start_pos, end_pos, \
                    recruiter_num, distance, fare_amount, reward_state, real_dis, move_dis]
            self.new_task_list.append(task)
            self.pending_schedules[task_id] = task

    def output_state(self, log_path, step=0):
        self.output_state_schedule(log_path, step)
        #self.output_state_action(log_path, step)

    def output_state_schedule(self, log_path, step=0):
        log_file = codecs.open(log_path, "a", "utf8")
        log_file.write("#states step:%s\n" % step)
        log_file.write("#current participants:\n")
        for p_id in self.participants:
            for item in self.participants[p_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("#current running_schedules: %d\n" % len(self.running_schedules))
        for task_id in self.running_schedules:
            for item in self.running_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("#current pending_schedules: %d\n" % len(self.pending_schedules))
        for task_id in self.pending_schedules:
            for item in self.pending_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("#current finished_schedules: %d\n" % len(self.finished_schedules))
        for task_id in self.finished_schedules:
            for item in self.finished_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("#reward:%s\n" % self.reward)
        log_file.write("#final reward:%s\n" % self.final_reward)
        log_file.write("\n")
        log_file.close()

    def output_state_action(self, log_path, step=0):
        log_file = codecs.open(log_path, "a", "utf8")
        log_file.write("#states step:%s\n" % step)
        log_file.write("#available participants: [")
        for p_id in self.participants:
            if self.participants[p_id][1] == ParticipantState["available"]:
                log_file.write("[%s,%s]\t" % (self.participants[p_id][0], self.participants[p_id][7]))
        log_file.write("]\n")
        log_file.write("#working participants: [")
        for p_id in self.participants:
            if self.participants[p_id][1] != ParticipantState["available"]:
                log_file.write("[%s,%s]\t" % (self.participants[p_id][0], self.participants[p_id][7]))
        log_file.write("]\n")
        log_file.write("#new tasks: [")
        for action in self.exe_actions:
            log_file.write("%s\t" % action)
        log_file.write("]\n")
        log_file.write("#reward:%s\n" % self.reward)
        log_file.write("#final reward:%s\n" % self.final_reward)
        log_file.write("\n")
        log_file.close()
