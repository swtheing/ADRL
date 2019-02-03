#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs
from enum import Enum
import random
import math


class TaskState(Enum):
    pending = 0
    picking = 1
    picked = 2
    finished = 3


class ParticipantState(Enum):
    available = 0
    working = 1
    unavailable = -1


class Trajectory():
    def __init__(self):
        self.trajectories = []
        self.ave_speed = 0.0

    def init_sampling(self, trajectory_data, sampling_size=10000):
        """
        input raw trajectory_data and sampling trajectories
        """
        raw_trajectories = random.sample(trajectory_data, sampling_size)
        self.trajectories = [x[2] for x in raw_trajectories]

    def get_distance(self, x1, y1, x2, y2):
        return pow((pow((x1 - x2), 2) + pow((y1 - y2), 2)), 0.5)

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
            # print start_x
            # print start_y
            # print end_x
            # print end_y
            distance = self.get_distance(start_x, start_y, end_x, end_y)
            sum_distance += distance
        # print sum_duration
        # print sum_distance
        self.ave_speed = sum_distance/sum_duration
        return self.ave_speed

    def yeild_rand_position(self):
        random.shuffle(self.trajectories)
        return self.trajectories[0]

    def move_simulate(self, start_pos, target_pos, speed, unit_time=300):
        x1, y1 = start_pos
        x2, y2 = target_pos
        x3, y3 = start_pos
        distance = self.get_distance(x1, y1, x2, y2)
        move_dis = speed * unit_time
        if move_dis > distance:
            # finish
            return target_pos, 0.0
        if x2 != x1:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - (k * x1)
            theta = math.atan(abs(x1 - x2) / abs(y1 - y2))
            delta_y = move_dis * math.cos(theta)
            if y1 <= y1 - delta_y <= y2:
                y3 = y1 - delta_y
            else:
                y3 = y1 + delta_y
            x3 = (y3 - b) / k
        else:
            delta_y = move_dis
            if y1 <= y1 - delta_y <= y2:
                y3 = y1 - delta_y
            else:
                y3 = y1 + delta_y
            x3 = x1
        new_pos = [x3, y3]
        return new_pos, distance - move_dis


class StateSimulator():
    def __init__(self):
        self.running_schedules = dict()
        self.pending_schedules = dict()
        self.finished_schedules = dict()
        self.participants = dict()

    def init_participants(self, trajactor, participants_num=10):
        for i in range(0, participants_num):
            participant_id = i
            start_pos = trajactor.yeild_rand_position()
            target_pos = trajactor.yeild_rand_position()
            cur_pos = start_pos
            remain_distance = trajactor.get_distance( \
                start_pos[0], start_pos[1], target_pos[0], target_pos[1])
            task_id = "DEFAULT"
            state = ParticipantState["available"]
            participant = [participant_id, state, task_id, \
                cur_pos, start_pos, target_pos, remain_distance]
            self.participants[participant_id] = participant

    def free_working_participant(self, participant_id, trajactor):
        participant_id, state, task_id, \
            cur_pos, start_pos, target_pos, remain_distance = self.participants[participant_id]
        target_pos = trajactor.yeild_rand_position()
        cur_pos = start_pos
        remain_distance = trajactor.get_distance( \
            start_pos[0], start_pos[1], target_pos[0], target_pos[1])
        task_id = "DEFAULT"
        state = ParticipantState["available"]
        self.participants[participant_id] = [participant_id, state, task_id, \
            cur_pos, start_pos, target_pos, remain_distance]

    def update_state(self, new_tasks, actions, trajector, unit_time=300):
        """
        2.update positon
        3.execute action
        1.update new tasks
        """
        self.update_position(trajector, unit_time)
        self.execute_action(actions, trajector)
        self.update_new_tasks(new_tasks)

    def update_position(self, trajector, unit_time=300):
        for p_id in self.participants:
            p_id, state, task_id, \
                cur_pos, start_pos, target_pos, remain_distance = self.participants[p_id]
            if state != ParticipantState["unavailable"]:
                new_pos, remain_distance = \
                    trajector.move_simulate(cur_pos, target_pos, trajector.ave_speed, unit_time)
                cur_pos = new_pos
                if remain_distance == 0: # finished
                    if state == ParticipantState["working"]: #已接单
                        if task_id in self.running_schedules:
                            task = self.running_schedules[task_id]
                            state = task[1]
                            if state == TaskState["picking"]: #接到人, refresh task
                                state = TaskState["picked"]   #开始执行订单
                                target_pos = task[6]
                                start_pos = task[5]
                                task[4] = start_pos
                                task[1] = state
                                self.running_schedules.pop(task_id)
                                self.running_schedules[task_id] = task
                                # continue current traj
                                self.participants[p_id][3] = cur_pos
                                self.participants[p_id][6] = remain_distance
                            elif state == TaskState["picked"]: #完成订单
                                self.finished_schedules[task_id] = task
                                self.finished_schedules[task_id][1] = TaskState["finished"]
                                self.running_schedules.pop(task_id)
                                self.free_working_participant(p_id, trajector)
                    else:
                        self.free_working_participant(p_id, trajector)
                else:
                    # continue random walk
                    self.participants[p_id][3] = cur_pos
                    self.participants[p_id][6] = remain_distance

    def execute_action(self, actions, trajactor):
        for action in actions:
            action_name, p_id, t_id = action
            if p_id in self.participants \
                    and t_id in self.pending_schedules \
                    and self.participants[p_id][1] == ParticipantState["available"] \
                    and self.pending_schedules[t_id][1] == TaskState["pending"]:
                # update participant
                p_id, state, p_task_id, \
                    cur_pos, start_pos, target_pos, remain_distance = self.participants[p_id]
                state = ParticipantState["working"]
                target_pos = self.pending_schedules[t_id][5]
                start_pos = cur_pos
                remain_distance = trajactor.get_distance(cur_pos[0], cur_pos[1], target_pos[0], target_pos[1])
                self.participants[p_id] = [p_id, state, t_id, \
                    cur_pos, start_pos, target_pos, remain_distance]
                # update task
                self.running_schedules[t_id] = self.pending_schedules[t_id]
                self.running_schedules[t_id][1] = TaskState["picking"]
                self.pending_schedules.pop(t_id)

    def update_new_tasks(self, new_tasks):
        """
        args:
        new_tasks:
        [[task_id,key_sec,
          time, [start_t, end_t] (year, month, day, hour, minute, second, unix_time, time_str)
          position, (start_x, end_x, start_y, end_y)
          passenger_num, distance, fare_amount]] 
        """
        for task_data in new_tasks:
            task_id = task_data[0]                                  #id
            state = TaskState["pending"]                            #state
            time = task_data[2]                                     #开始and结束时间
            participant_id = -1
            cur_pos = []                                            #当前位置
            start_pos = task_data[3][0:2]                           #起点
            end_pos = task_data[3][2:4]                             #终点
            recruiter_num, distance, fare_amount = task_data[4:7]   #乘客人数，距离，花费
            pre_progress_rate = 0.0                                 #预完成进度
            progress_rate = 0.0                                     #完成进度
            task = [task_id, state, participant_id, cur_pos, start_pos, end_pos, \
                    recruiter_num, distance, fare_amount, pre_progress_rate, progress_rate]
            self.pending_schedules[task_id] = task

    def output_state(self, log_path, step=0):
        log_file = codecs.open(log_path, "a", "utf8")
        log_file.write("\n\nstates step:%s\n" % step)
        log_file.write("current participants:\n")
        for p_id in self.participants:
            for item in self.participants[p_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("\ncurrent running_schedules: %d\n" % len(self.running_schedules))
        for task_id in self.running_schedules:
            for item in self.running_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("\ncurrent pending_schedules: %d\n" % len(self.pending_schedules))
        for task_id in self.pending_schedules:
            for item in self.pending_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.write("\ncurrent finished_schedules: %d\n" % len(self.finished_schedules))
        for task_id in self.finished_schedules:
            for item in self.finished_schedules[task_id]:
                log_file.write("%s\t" % item)
            log_file.write("\n")
        log_file.close()
