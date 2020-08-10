#
import random
import numpy as np
from Data_Generator.env_trj import *
from Data_Generator.env_trj.state_generator import *

class Greedy_OPT():
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
        observation = self.env.reset(True)
        raw_rewards = []
        fare_std_rewards = []
        time_std_rewards = []
        fare_means = []
        disc_means = []

        for step_n in range(total_step):
            # Nearest First, else random
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
            for task in observation.new_task_list:
                taskid = task[0]
                if taskid in cur_task_set:
                    continue
                if len(cur_pid_set) < len(available_pid_list): #len(available_pid_list)>0
                    t_start_pos = observation.pending_schedules[taskid][3][0:2]
                    action_taskid = taskid
                    candidate_pid = -1
                    # random
                    rand_t = random.random()
                    distance = 0.0
                    if rand_t < self.config.random_prob:
                        candidate_pid = random.randint(1, len(observation.participants)) # start from 1
                    else:
                        # greedy
                        min_dis = -1.0
                        for pid in available_pid_list:
                            if pid in cur_pid_set:
                                continue
                            p_cur_pos = observation.participants[pid][3]
                            distance = observation.trajector.get_distance( \
                                t_start_pos[0], t_start_pos[1], p_cur_pos[0], p_cur_pos[1])
                            #print "DISTANCE %s %s %s " % (pid, taskid, distance)
                            if distance < min_dis or min_dis < 0:
                                candidate_pid = pid
                                min_dis = distance
                                pick_flag = 1
                        if pick_flag == 0:
                            continue
                    action_pid = candidate_pid
                    action_taskid = taskid
                    cur_pid_set.add(action_pid)
                    cur_task_set.add(action_taskid)
                    action = ["pick", action_pid, action_taskid]
                    action_idlist.append(action_pid)
                    actions.append(action)

                if taskid not in cur_task_set and len(available_pid_list) > 0:
                    action_pid = random.randint(1, len(observation.participants)) # start from 1
                    action_taskid = taskid
                    action = ["pick", action_pid, action_taskid]
                    action_idlist.append(action_pid)
                    actions.append(action)
            print action_idlist
            # print len(observation.participants)
            # print self.config.max_task_size
            # for i in range(len(observation.participants), self.config.max_task_size):
            #     print i
            #     action_idlist.append(0)
            
            length = len(action_idlist)
            for i in range(length, self.config.max_task_size):
                action_idlist.append(0)
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
            print "greedy opt result:"
            print "ave_reward:%s, max_reward:%s, min_reward:%s" % (ave_reward, max_reward, min_reward)
            print "mean_raw_reward:%s, mean_fare_std_r:%s, mean_time_std_r:%s, mean_dis_cost:%s, mean_fare:%s" \
                % (np.mean(raw_rewards), np.mean(fare_std_rewards), np.mean(time_std_rewards), \
                    np.mean(disc_means), np.mean(fare_means))
            print "total_step:%s, total_episode:%s, total_reward:%s"\
                % (total_step, total_episode, total_reward)
        else:
            print "None episode finished"
        return ave_reward, total_pending_time/total_episode, total_fare_amount/total_episode


                            # if observation.participants[pid][1] == ParticipantState["available"]:
                            # par_pos = observation.participants[pid][3]  # availabel->cur pos
                        # elif observation.participants[pid][1] == ParticipantState["working"]:
                        #     par_pos = observation.participants[pid][5]  # working->target pos


