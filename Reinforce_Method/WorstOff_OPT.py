
import random
import numpy as np
from Data_Generator.env_trj import *
from Data_Generator.env_trj.state_generator import *

class WorstOff():
    def __init__(self, config, game_name, env):
        #model = Perceptron(game_name, None, config, "MSE", config.pre)
        #Reinforce_Suite.__init__(self, config, model, env)
        self.env = env
        self.config = config

    def run_test(self, total_step, pre_process=True, is_test=True):
        if pre_process and is_test:
            return self.pre_process_test(total_step, is_test)
        else:
            return self.run_test_raw(total_step)

    def pre_process_test(self, total_step, is_test=True):
        total_reward = 0.0
        total_episode = 0
        total_pending_time = 0.0
        total_fare_amount = 0.0
        ave_reward = 0.0
        max_reward = -99999999.0
        min_reward = 99999999.0
        raw_rewards = []
        fare_std_rewards = []
        time_std_rewards = []
        fare_means = []
        disc_means = []
        observation = self.env.reset(True)

        for step_n in range(total_step):
            print "STEP %d" % (step_n + 1)
            # random
            action_taskid = -1
            action_pid = -1
            cur_pid_set = set()
            cur_task_set = set()
            #actions = observation.pending_actions
            action_idlist = []
            actions = []
            available_pid_list = []
            for pid in observation.participants:
                if observation.participants[pid][1] == ParticipantState["available"]:
                    available_pid_list.append(pid)

            # find best task
            best_task_id = -1
            best_fare = -1.0
            for task in observation.new_task_list:
                taskid = task[0]
                task_fare = task[10]
                if taskid in cur_task_set:
                    continue
                if task_fare > best_fare:
                    best_fare = task_fare
                    best_task_id = taskid
            print "best fare:%s, tid:%s" % (best_fare, best_task_id)

            # find best match
            if best_task_id != -1:
                # find worst par
                t_start_pos = observation.pending_schedules[best_task_id][3][0:2]
                min_fare = 999999999.0
                fare = 0.0
                pick_flag = 0
                pid_index = -1
                for i in range(len(available_pid_list)):
                    pid = available_pid_list[i]
                    if pid not in cur_pid_set:
                        if pid in observation.participant_fare:
                            fare = observation.participant_fare[pid]
                        else:
                            fare = 0.0
                        p_cur_pos = observation.participants[pid][3]
                        distance = observation.trajector.get_distance( \
                            t_start_pos[0], t_start_pos[1], p_cur_pos[0], p_cur_pos[1])
                        #print "DISTANCE %s %s %s " % (pid, taskid, distance)
                        if distance > best_fare:
                            continue
                        if fare <= min_fare:
                            candidate_pid = pid
                            min_fare = fare
                            pick_flag = 1
                            pid_index = i
                            print "min FARE p%s t%s %s " % (pid, best_task_id, fare)
                if pick_flag == 1:
                    action_pid = candidate_pid
                    action_taskid = best_task_id
                    cur_pid_set.add(action_pid)
                    cur_task_set.add(action_taskid)
                    action = ["pick", action_pid, action_taskid]
                    print action
                    actions.append(action)
                    #action_idlist.append(action_pid)
                    del available_pid_list[pid_index]

            # rest random
            for task in observation.new_task_list:
                taskid = task[0]
                if taskid in cur_task_set:
                    continue
                picked_flag = 0
                if len(available_pid_list) > 0:
                    index = random.randint(0, len(available_pid_list)-1) # rand pid
                    action_pid = available_pid_list[index]
                    if action_pid not in cur_pid_set:
                        action_taskid = taskid
                        cur_pid_set.add(action_pid)
                        cur_task_set.add(action_taskid)
                        action = ["pick", action_pid, action_taskid]
                        #action_idlist.append(action_pid)
                        actions.append(action)
                        del available_pid_list[index]

            for task in observation.new_task_list:
                taskid = task[0]
                pick_flag = 0
                for action in actions:
                    info, action_pid, action_taskid = action
                    if action_taskid == taskid:
                        action_idlist.append(action_pid)
                        pick_flag = 1
                if pick_flag == 0:
                    action_idlist.append(0)
            print actions
            print action_idlist

            new_observation, reward, done, info = self.env.step(action_idlist, True)
            raw_reward, reward_fare_std, reward_time_std, mean_time_cost, std_time_cost, mean_fare_amount, std_fare_amount, finished_task_num, mean_dis_cost, std_dis_cost, mean_finish, std_finish = info
            observation = new_observation
            if done:
                total_episode += 1
                total_reward += reward
                total_pending_time += mean_time_cost
                total_fare_amount += mean_fare_amount
                if reward > max_reward:
                    max_reward = reward
                if reward < min_reward:
                    min_reward = reward
                print "episode: %d, reward: %s" % (total_episode, reward)
                print "finished task %d" % (len(observation.finished_schedules))
                observation = self.env.reset(True)

                raw_rewards.append(raw_reward)
                fare_std_rewards.append(reward_fare_std)
                time_std_rewards.append(reward_time_std)
                fare_means.append(mean_fare_amount)
                disc_means.append(mean_dis_cost)
        if total_episode > 0:
            ave_reward = total_reward / total_episode
            print "worstoff result:"
            print "ave_reward:%s, max_reward:%s, min_reward:%s" % (ave_reward, max_reward, min_reward)
            print "mean_raw_reward:%s, mean_fare_std_r:%s, mean_time_std_r:%s, mean_dis_cost:%s, mean_fare:%s" \
                % (np.mean(raw_rewards), np.mean(fare_std_rewards), np.mean(time_std_rewards), np.mean(disc_means), np.mean(fare_means))
            print "total_step:%s, total_episode:%s, total_reward:%s"\
                % (total_step, total_episode, total_reward)
        else:
            print "None episode finished"
        return ave_reward, total_pending_time/total_episode, total_fare_amount/total_episode

