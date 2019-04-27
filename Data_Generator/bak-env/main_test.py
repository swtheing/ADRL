#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs
import data_load
import task_generator
import state_generator
from copy import deepcopy

def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
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

if __name__ == "__main__":
    data_loader = data_load.DataLoader()
    aim_day_num = 1
    data_loader.get_merge_task(aim_day_num) # 读取yellow 数据
    # zip_file = codecs.open("zip_data.txt", "w", "utf8")
    # data_loader.output_merge_data(zip_file)
    # zip_file.close()
    print "task generation"
    tg = task_generator.TaskGenerator()
    tg.gen_task_list(data_loader.zip_data) # 按分布采样生成task列表
    simulator1 = state_generator.StateSimulator()
    print "trajectory sampling"
    data_loader.get_trajectories() # 读取uber数据
    print "trajectory size: %d" % len(data_loader.trajectory_data)
    simulator1.trajector.init_sampling(data_loader.trajectory_data, 100)  # 采样生成路线数据
    ave_speed = simulator1.trajector.set_ave_speed(0.00005)
    print "ave speed: %s" % ave_speed
    data_loader.reset() # clear memeory

    print "task sampling"
    episode_task_num = 40

    # clear log
    log_file = codecs.open("../random.state.log", "w", "utf8")
    log_file.close()
    g_log_file = codecs.open("../greedy.state.log", "w", "utf8")
    g_log_file.close()

    simulator1.init_participants(5) # 初始化20个taxi
    simulator2 = deepcopy(simulator1)

    task_samples = tg.task_generation(episode_task_num)
    print "step:0, task_num:%s" % len(task_samples)
    actions1 = []
    actions2 = []
    simulator1.update_state(task_samples, actions1)
    simulator1.output_state("../random.state.log", 0)
    simulator2.update_state(task_samples, actions2)
    simulator2.output_state("../greedy.state.log", 0)
    
    for step in range(1, 20):
        task_samples = tg.task_generation(episode_task_num)
        print "step:%s, task_num:%s" % (step, len(task_samples))

        # random
        for pid in simulator1.participants:
            if simulator1.participants[pid][1] == state_generator.ParticipantState["available"]:
                action_pid1 = pid
                break
        for taskid in simulator1.pending_schedules:
            action_task1 = taskid
            break
        action1 = ["pick", action_pid1, action_task1]
        actions1 = [action1]
        simulator1.update_state(task_samples, actions1)
        simulator1.output_state("../random.state.log", step)
    
        # greedy choose, every step chose one
        for taskid in simulator2.pending_schedules:
            t_start_pos = simulator2.pending_schedules[taskid][3][0:2]
            min_dis = -1.0
            for pid in simulator2.participants:
                if simulator2.participants[pid][1] == state_generator.ParticipantState["available"]:
                    p_start_pos = simulator2.participants[pid][3]
                    distance = simulator2.trajector.get_distance(t_start_pos[0], t_start_pos[1], p_start_pos[0], p_start_pos[1])
                    if distance < min_dis or min_dis < 0:
                        action_pid2 = pid
                        min_dis = distance
            action_task2 = taskid
            break

        action2 = ["pick", action_pid2, action_task2]
        actions2 = [action2]
        simulator2.update_state(task_samples, actions2)
        simulator2.output_state("../greedy.state.log", step)




