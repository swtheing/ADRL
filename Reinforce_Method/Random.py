#Off-Policy Learning and TD(1) Learining
import random
import time
# from Reinforce_Suite import *
# from Perceptron import *
# from Cnn import *
# from gym.envs.classic_control import rendering
from Data_Generator.env_trj import *
from Data_Generator.env_trj.state_generator import *

class Random():
    def __init__(self, config, game_name, env):
        #model = Perceptron(game_name, None, config, "MSE", config.pre)
        #Reinforce_Suite.__init__(self, config, model, env)
        self.env = env
        self.config = config

    def run_test(self, total_step, pre_process=True, is_test=True):
        if pre_process and is_test:
            self.pre_process_test(total_step, is_test)
        else:
            self.run_test_raw(total_step)

    # def run_test_raw(self, total_step):
    #     total_reward = 0.0
    #     total_episode = 0
    #     max_reward = 0.0
    #     min_reward = 99999999.0
    #     observation = self.env.reset(True)
    #     for i in range(total_step):
    #         # random
    #         action_taskid = -1
    #         action_pid = -1
    #         cur_pid_set = set()
    #         cur_task_set = set()
    #         actions = []
    #         for taskid in observation.pending_schedules:
    #             pick_flag = 0
    #             if taskid in cur_task_set:
    #                 continue
    #             action_taskid = taskid
    #             for pid in observation.participants:
    #                 if (pid not in cur_pid_set) and observation.participants[pid][1] == ParticipantState["available"]:
    #                     action_pid = pid
    #                     pick_flag = 1
    #                     break
    #             if pick_flag == 0:
    #                 continue
    #             cur_pid_set.add(action_pid)
    #             cur_task_set.add(action_taskid)
    #             action = ["pick", action_pid, action_taskid]
    #             print action
    #             actions.append(action)
    #         new_observation, reward, done, info = self.env.step_raw(actions)
    #         observation = new_observation
    #         if done:
    #             total_episode += 1
    #             total_reward += reward
    #             observation = self.env.reset(True)
    #             print "episode: %d, reward: %s" % (total_episode, reward)
    #     ave_reward = total_reward / total_episode
    #     print "ave_reward: %s" % ave_reward

    def pre_process_test(self, total_step, is_test=True):
        total_reward = 0.0
        total_episode = 0
        max_reward = 0.0
        min_reward = 99999999.0
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
                #if observation.participants[pid][1] == ParticipantState["available"]:
                available_pid_list.append(pid)
            for task in observation.new_task_list:
                taskid = task[0]
                action_pid = random.randint(1, len(observation.participants)) # start from 1
                action = ["pick", action_pid, action_taskid]
                action_idlist.append(action_pid)

            """
            for task in observation.new_task_list:
                taskid = task[0]
                if taskid in cur_task_set:
                    continue
                if len(cur_pid_set) < len(available_pid_list): #len(available_pid_list)>0
                    # random
                    random.shuffle(available_pid_list)
                    candidate_pid = available_pid_list[0]
                    get_pid_flag = False
                    while candidate_pid in cur_pid_set:
                        random.shuffle(available_pid_list)
                        candidate_pid = available_pid_list[0]
                    action_pid = candidate_pid
                    action_taskid = taskid
                    cur_pid_set.add(action_pid)
                    cur_task_set.add(action_taskid)
                    action = ["pick", action_pid, action_taskid]
                    action_idlist.append(action_pid)
                    actions.append(action)
                if taskid not in cur_task_set:
                    action_pid = random.randint(1, len(observation.participants)) # start from 1
                    # print "rand"
                    # print action_pid
                    action_taskid = taskid
                    action = ["pick", action_pid, action_taskid]
                    action_idlist.append(action_pid)
                    actions.append(action) 
            length = len(action_idlist)
            for i in range(length, self.config.max_task_size):
                # action_idlist.append(0)
                random.shuffle(available_pid_list)
                action_idlist.append(available_pid_list[0])
            """
            new_observation, reward, done, info = self.env.step(action_idlist, is_test)
            observation = new_observation
            if done:
                total_episode += 1
                total_reward += reward
                if reward > max_reward:
                    max_reward = reward
                if reward < min_reward:
                    min_reward = reward
                print "episode: %d, reward: %s" % (total_episode, reward)
                print "finished task %d" % (len(observation.finished_schedules))
                observation = self.env.reset(True)
        if total_episode > 0:
            ave_reward = total_reward / total_episode
            # print "ave_reward: %s" % (ave_reward / self.config.max_step)
            # print "max_reward: %s" % (max_reward / self.config.max_step)
            # print "min_reward: %s" % (min_reward / self.config.max_step)
            print "ave_reward: %s" % (ave_reward)
            print "max_reward: %s" % (max_reward)
            print "min_reward: %s" % (min_reward)
        else:
            print "None episode finished"

