#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs
import random

class TaskGenerator():
    def __init__(self):
        self.step_counter = -1
        self.total_task_num = 0
        self.time_list = []
        self.task_distribution = dict()
        self.task_candidates = dict()

    def task_generation(self, episode_task_num=10000):
        self.step_counter += 1
        if self.step_counter >= len(self.time_list):
            self.step_counter = 0
        key_t = self.time_list[self.step_counter]
        task_samples_num = int(self.task_distribution[key_t] * episode_task_num)
        task_samples = random.sample(self.task_candidates[key_t], task_samples_num)
        return task_samples

    def gen_task_list(self, zip_task_data, unit_time=300, key_day=0):
        """
        args: statistic task data for one day
        returns: extract task
        """
        task_data = zip_task_data[key_day][1]
        self.zip_task_analysis(task_data, unit_time)
        # for key_t in self.time_list:
        #     print "%s\t%s\t%s" % (key_t, self.task_distribution[key_t], len(self.task_candidates[key_t]))

    def zip_task_analysis(self, task_data, unit_time=300):
        """
        statistic task data for one day
        every unit time
        return [time, task_id, task]
        """
        start_unix_time = task_data[0][0]
        end_unix_time = task_data[-1][0]
        assert(start_unix_time < end_unix_time)
        start_t = 0
        end_t = -1
        for index in range(len(task_data)):
            task_t = task_data[index][0]
            if start_t <= task_t and task_t < end_t:
                self.task_distribution[start_t] += 1.0
                task_id = "%s_%05d" % (task_t, index)
                task = [task_id] + task_data[index]
                self.task_candidates[start_t].append(task)
                self.total_task_num += 1
            elif task_t >= end_t:
                end_t = task_t if end_t <= start_t else end_t
                start_t = end_t
                end_t = start_t + unit_time
                self.time_list.append(start_t)
                self.task_candidates[start_t] = []
                self.task_distribution[start_t] = 0.0
        for key_t in self.task_distribution:
            self.task_distribution[key_t] = self.task_distribution[key_t]/self.total_task_num


