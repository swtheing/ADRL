#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs

class DataLoader():
    def __init__(self):
        self.task_data_path = "../data/tasks"
        self.trajectory_data_path = "../data/trajectory"
        self.raw_data = []
        self.zip_data = []
        self.trajectory_data = []

    def __init__(self, task_data_path, trajectory_data_path):
        self.task_data_path = task_data_path
        self.trajectory_data_path = trajectory_data_path
        self.raw_data = []
        self.zip_data = []
        self.trajectory_data = []

    def reset(self):
        del self.raw_data
        del self.zip_data
        del self.trajectory_data
        self.raw_data = []
        self.zip_data = []
        self.trajectory_data = []

    def reset_raw_data(self):
        del self.raw_data
        self.raw_data = []

    def get_merge_task(self, aim_day_num):
        """
        把当前全部task数据压缩至aim_day_num天并返回
        """
        self.load_task_static(self.task_data_path)
        self.merge_data(aim_day_num)

    def get_trajectories(self):
        """
        读取Uber路线数据
        """
        self.load_trajectory(self.trajectory_data_path)

    def parse_time_str_uber(self, time_str, date_spliter="/"):
        """
        parse time format
        """
        time_str = time_str.strip()
        terms = time_str.split(" ")
        if len(terms) != 2:
            return []
        date = terms[0].split(date_spliter)
        time = terms[1].split(":")
        if len(date) != 3 or len(time) != 3:
            return []
        year = int(date[2])
        month = int(date[0])
        day = int(date[1])
        hour = int(time[0])
        minute = int(time[1])
        second = int(time[2])
        unix_time = (year - 1970) * 31556736 + month * 2629743 + day * 86400 \
                     + hour * 3600 + minute * 60 + second
        return [year, month, day, hour, minute, second, unix_time, time_str]

    def parse_time_str(self, time_str, date_spliter="-"):
        """
        parse time format
        """
        time_str = time_str.strip()
        terms = time_str.split(" ")
        if len(terms) != 2:
            return []
        date = terms[0].split(date_spliter)
        time = terms[1].split(":")
        if len(date) != 3 or len(time) != 3:
            return []
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])
        hour = int(time[0])
        minute = int(time[1])
        second = int(time[2])
        unix_time = (year - 1970) * 31556736 + month * 2629743 + day * 86400 \
                     + hour * 3600 + minute * 60 + second
        return [year, month, day, hour, minute, second, unix_time, time_str]

    def load_task_static(self, tasks_dir):
        """
        load raw task data
        """
        files= os.listdir(tasks_dir)
        for file in files:
            if "trip" not in file:
                continue
            f_path = tasks_dir + "/" + file
            if os.path.isdir(f_path): 
                continue
            print f_path
            file = codecs.open(f_path, "r", "utf8")
            for line in file:
                try:
                    terms = line.strip().split(",")
                    if len(terms) != 19:
                        continue
                    if "_" in terms[1] or len(terms[1]) == 0:
                        continue
                    start_t = self.parse_time_str(terms[1])
                    end_t = self.parse_time_str(terms[2])
                    start_x = float(terms[5])
                    start_y = float(terms[6])
                    end_x = float(terms[9])
                    end_y = float(terms[10])
                    time = [start_t, end_t]
                    position = [start_x, start_y, end_x, end_y]
                    passenger_num = float(terms[3])
                    distance = float(terms[4])
                    fare_amount = float(terms[12])
                    if len(start_t) == 0 or len(end_t) == 0:
                        continue
                    if start_x == 0 or end_x == 0 or \
                            start_y == 0 or end_y == 0 \
                            or distance == 0 or fare_amount == 0:
                        continue
                    self.raw_data.append([time, position, passenger_num, distance, fare_amount])
                except Exception as e:
                    print "[Data error]: %s, %s" % (e, line)
            file.close()

    def merge_data(self, aim_day_num):
        """
        zip raw_data to one day
        Returns: 
          a dict mapped by day
          [[key_day,[key_sec, 
          start_t, end_t, (year, month, day, hour, minute, second, unix_time, time_str)
          position, (start_x, end_x, start_y, end_y)
          passenger_num, distance, fare_amount]] 
        """
        data_dic = dict()
        for data in self.raw_data:
            day = data[0][0][2]
            key_day = day % aim_day_num  # merge days
            key_sec =  data[0][0][6]
            new_data = [key_sec] + data
            if key_day in data_dic:
                data_dic[key_day].append(new_data)
            else:
                data_dic[key_day] = [new_data]
        print "merge_data_num: %d" % len(self.raw_data)
        for key_day in data_dic:
            data_list = data_dic[key_day]
            print "%d key_day_num: %d" % (key_day, len(data_list))
            data_sort = sorted(data_list, key=lambda d: d[0], reverse=False)
            data_dic[key_day] = data_sort
        sortedlist = sorted(data_dic.iteritems(), key=lambda d: d[0], reverse=False)
        print "total key list:%d" % len(sortedlist)
        self.zip_data = sortedlist

    def output_merge_data(self, out_file):
        for data in self.zip_data:
            assert(len(data) == 2)
            key_day = data[0]
            data_list = data[1]
            for item in data_list:
                assert(len(item) == 10)
                key_sec, time, pos, passenger_num, distance, fare_amount = item
                out_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        % (key_day, key_sec, time[0][-1], time[1][-1], 
                            passenger_num, distance, fare_amount))

    def load_trajectory(self, trajectory_dir):
        """
        load trajectory from uber dataset
        """
        files = os.listdir(trajectory_dir)
        for file in files:
            if "uber" not in file:
                continue
            f_path = trajectory_dir + "/" + file
            if os.path.isdir(f_path): 
                continue
            print f_path
            file = codecs.open(f_path, "r", "utf8")
            for line in file:
                if len(self.trajectory_data) > 10000:
                    break
                try:
                    terms = line.strip().split(",")
                    if len(terms) != 4:
                        continue
                    if "Date" in terms[0]:
                        continue
                    time = self.parse_time_str_uber(terms[0][1:-1], "/")
                    if len(time) == 0:
                        continue
                    pos_y = float(terms[1])
                    pos_x = float(terms[2])
                    position = [pos_x, pos_y]
                    participant_id = terms[3][1:-1]
                    self.trajectory_data.append([participant_id, time, position])
                except Exception as e:
                    print "[Data error]: %s, %s" % (e, line)
            file.close()

