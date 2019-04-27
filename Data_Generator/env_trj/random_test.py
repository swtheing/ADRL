#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import codecs
import env_lbs
import data_load
import task_generator
import state_generator

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
    task_list = tg.gen_task_list(data_loader.zip_data) # 按分布采样生成task列表
    simulator = state_generator.StateSimulator()
    print "trajectory sampling"
    data_loader.get_trajectories() # 读取uber数据
    print "trajectory size: %d" % len(data_loader.trajectory_data)
    simulator.trajector.init_sampling(data_loader.trajectory_data, 100)  # 采样生成路线数据
    ave_speed = simulator.trajector.set_ave_speed(0.00005)
    print "ave speed: %s" % ave_speed
    data_loader.reset() # clear memeory

    print "task sampling"
    episode_task_num = 40

    # clear log
    log_file = codecs.open("../state.log", "w", "utf8")
    log_file.close()

    simulator.init_participants(5) # 初始化20个taxi
    #new_pos, remain_distance = tj.move_simulate([5.0,1.0], [1.0,2.0], 1.414, 0.5, 2)
    # print new_pos[0]
    # print new_pos[1]
    # print remain_distance

    task_samples = tg.task_generation(episode_task_num)
    print "step:0, task_num:%s" % len(task_samples)
    actions = []
    simulator.update_state(task_samples, actions)
    simulator.output_state("../state.log", 0)
    
    for step in range(1, 20):
        task_samples = tg.task_generation(episode_task_num)
        print "step:%s, task_num:%s" % (step, len(task_samples))
        for pid in simulator.participants:
            if simulator.participants[pid][1] == state_generator.ParticipantState["available"]:
                action_pid = pid
                break
        for taskid in simulator.pending_schedules:
            action_task = taskid
            break
        action = ["pick", action_pid, action_task]
        actions = [action]
        simulator.update_state(task_samples, actions)
        simulator.output_state("../state.log", step)
    




