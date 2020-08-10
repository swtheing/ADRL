#REINFORCE, MC, Policy Gradient
import random
import time
from Reinforce_Suite import *
from Model.Perceptron import *
from Model.Linear import *
from Model.Cnn import *
from Model.Gaussian import *
from Model.Trans_Ptr import *
from Data_Generator.env_trj import *
from Data_Generator.env_trj.state_generator import *

class Actor_Critic(Reinforce_Suite):
    def __init__(self, config, game_name,  env):
        if config.model == "DNN":
            Actor = Perceptron(game_name, None, config, "CE")
            self.conti_act = False
            self.multi_act = False
        elif config.model == "Gaussian":
            Actor = Gaussian(game_name, None, config, None)
            self.conti_act = True
            self.multi_act = False
        elif config.model == "CNN":
            Actor = Cnn(game_name, None, config, "CE")
            self.conti_act = False
            self.multi_act = False
        elif config.model == "TranPtr":
            Actor = Trans_Ptr(game_name, None, config, "CE", "Actor")
            self.Critic = Trans_Ptr(game_name, None, config, "MSE", "Critic")
            self.conti_act = False
            self.multi_act = True
        Reinforce_Suite.__init__(self, config, Actor, env)
        self.replay_match = config.replay_match
        self.replay_size = config.replay_size
        self.observe_id = config.observe_id
        self.on_policy = config.on_policy
        self.replay_switch = config.replay_switch
        self.task_mask = config.task_mask
        self.replay_obs = []
        self.replay_act = []
        self.replay_rew = []
        self.replay_val = []
        self.replay_done = []
        self.replay_next = []
        self.base_v = 0.0
        self.sum_step = 0
        self.viewer = None


    def Gen_Batch_Data(self, policy, epoch_num):
        batchs = []
        for epoch in range(epoch_num):
            samples = random.sample(range(len(self.replay_obs)), self.model.batch_size)
            samples_obs = [self.replay_obs[i] for i in samples]
            #bug
            #samples_act = [self.replay_act[i] - 1 for i in samples]
            samples_act = [self.replay_act[i] for i in samples]
            samples_next =[self.replay_next[i] for i in samples]
            samples_epr = []
            samples_val = [self.replay_val[i] for i in samples]
            for i in samples:
                if self.replay_done[i]:
                    #print self.replay_rew[i]
                    samples_epr.append(self.replay_rew[i])
                else:
                    for j in range(i+1, len(self.replay_obs)):
                        #todo: add a param
                        if self.replay_rew[j] != 0.0:
                            samples_epr.append(self.replay_rew[j])
                            break
            tup = (samples_obs, samples_act, samples_epr, samples_val, samples_next)

            batchs.append(tup)
        return batchs

    def Eq_Action(self, a1, a2):
        print len(a1)
        for i in range(len(a1)):
            if a1[i] != a2[i] and a1[i] != -1:
                return 0
        return 1
                
    def Get_Data(self, policy):
        observation = self.env.reset()
        match = 0
        match_equal = 0
        over_reward = 0
        max_reward = -1000000.0
        min_reward = 1000000.0
        match_rerward = 0.0
        show_flag = 1

        if not self.replay_switch:
            self.replay_obs = []
            self.replay_act = []
            self.replay_rew = []
            self.replay_val = []
            self.replay_done = []
            self.replay_next = []

        if len(self.replay_obs) == self.replay_size:
            del self.replay_obs[0]
            del self.replay_done[0]
            del self.replay_next[0]
            del self.replay_rew[0]
            del self.replay_act[0]
            del self.replay_val[0]

        self.replay_obs.append(observation)
        while True:
            if self.task_mask == 1:
                gr_action = self.Greedy_action_mask(observation)
            else:
                gr_action = self.Greedy_action(observation)
            #gr_action = self.Greedy_action(observation)
            print "greedy_action:" + str(gr_action)
            action, _, Q_debug = policy.action_sel(observation, max_sel = False, continues = self.conti_act, multi_act = self.multi_act)
            _, val = self.Critic.test_model([observation])
            print "learn_action:" + str(action)
            match_equal += self.Eq_Action(gr_action, action)
             
            if len(self.replay_obs) > self.replay_size:
                del self.replay_obs[0]
                del self.replay_done[0]
                del self.replay_next[0]
                del self.replay_rew[0]
                del self.replay_act[0]
                del self.replay_val[0]
            #replay strategy
            # if self.observe_id < len(self.replay_obs):
            #     self.observe_picture = self.replay_obs[self.observe_id][25:,:,:]
            #     if (observation[25:,:,:] == self.observe_picture).all():
            #         if self.viewer is None:
            #             self.viewer = rendering.SimpleImageViewer()
            #         if show_flag == 1:
            #             self.viewer.imshow(observation[25:,:,:])
            #             show_flag = 0
            #         print "observe id: {}, action: {}, Q: {}".format(self.observe_id, action, Q_debug)
                #raw_input("trace image is here (Enter go): ");
            #action = [self.Greedy_action(observation) + 1]
            observation, reward, done, info = self.env.step(action)
            self.replay_rew.append(reward)
            self.replay_val.append(val)
            self.replay_done.append(done)
            self.replay_act.append(action)
            over_reward += reward
            match_rerward += reward
            if not done:
                self.replay_next.append(observation)
                self.replay_obs.append(observation)
            else:
                if match_rerward > max_reward:
                    max_reward = match_rerward
                elif match_rerward < min_reward:
                    min_reward = match_rerward
                match_rerward = 0
                self.replay_next.append(observation)
                match += 1
                if match == self.replay_match:
                    print "eq_match:" + str(float(match_equal) / self.replay_match)
                    return over_reward / self.replay_match, max_reward, min_reward
                observation = self.env.reset()
                self.replay_obs.append(observation)
    
    def distance(self, gps_1, gps_2):
        return pow((pow((gps_1[0] - gps_2[0]), 2) + pow((gps_1[1] - gps_2[1]), 2)), 0.5)


    def Greedy_action_mask(self, obs):
        par_f = obs[0]
        task_f = obs[1]
        # random
        action_taskid = -1
        action_pid = -1
        cur_pid_set = set()
        cur_task_set = set()
        #actions = observation.pending_actions
        action_idlist = []
        actions = []
        available_pid_list = []
        # print "task num:"
        # print len(task_f)
        for index in range(len(par_f)):
            participant = par_f[index]
            pid = index + 1
            available_pid_list.append(pid)
        for index in range(0, len(task_f)):
            if task_f[index][0] == 0:
                continue
            taskid = index + 1
            # print "taskid"
            # print taskid
            if taskid in cur_task_set:
                continue
            if len(cur_pid_set) >= len(available_pid_list): #len(available_pid_list)>0
                continue
            task = task_f[index]
            t_start_pos = [task[0], task[1]]
            action_taskid = taskid
            candidate_pid = -1

            # random
            rand_t = random.random()
            if rand_t <= -1.0:
                candidate_pid = random.randint(1, len(par_f)) # start from 1
            else:
                # greedy
                min_dis = -1.0
                for pid in available_pid_list:
                    if pid not in cur_pid_set:
                        p_start_pos = [par_f[pid-1][0], par_f[pid-1][1]]
                        dist = self.distance(t_start_pos, p_start_pos)
                        if dist < min_dis or min_dis < 0:
                            candidate_pid = pid
                            min_dis = dist
                            pick_flag = 1
                if pick_flag == 0:
                    continue
            action_pid = candidate_pid
            action_taskid = taskid
            cur_pid_set.add(action_pid)
            cur_task_set.add(action_taskid)
            action = ["pick", action_pid, action_taskid]
            # print action
            action_idlist.append(action_pid)
            actions.append(action)

            if taskid not in cur_task_set:
                action_pid = random.randint(1, len(par_f)) # start from 1
                action_taskid = taskid
                action = ["pick-rand", action_pid, action_taskid]
                # print action
                action_idlist.append(action_pid)
                actions.append(action)

        length = len(action_idlist)
        for i in range(length, len(task_f)):
            action_idlist.append(0)
        return action_idlist

    def Greedy_action(self, obs):
        par_f = obs[0]
        task_f = obs[1]
        # random
        action_taskid = -1
        action_pid = -1
        cur_pid_set = set()
        cur_task_set = set()
        #actions = observation.pending_actions
        action_idlist = []
        actions = []
        available_pid_list = []
        for index in range(len(par_f)):
            participant = par_f[index]
            pid = index + 1
            p_state = int(participant[0])
            #if p_state == ParticipantState["available"]:
            available_pid_list.append(pid)
        for index in range(len(task_f)):
            taskid = index + 1
            task = task_f[index]
            task_p = float(task[3])
            if task_p == 0:
                continue
            t_state = int(task[0])
            if t_state != TaskState["pending"]:
                continue
            if taskid in cur_task_set:
                continue
            if len(cur_pid_set) < len(available_pid_list): #len(available_pid_list)>0
                t_start_pos = [task[3], task[4]]
                action_taskid = taskid
                candidate_pid = -1

                # random
                rand_t = random.random()
                if rand_t <= -1.0:
                    candidate_pid = random.randint(1, len(par_f)) # start from 1
                else:
                    # greedy
                    min_dis = -1.0
                    for pid in available_pid_list:
                        if pid not in cur_pid_set:
                            p_start_pos = [par_f[pid-1][3], par_f[pid-1][4]]
                            dist = self.distance(t_start_pos, p_start_pos)
                            if dist < min_dis or min_dis < 0:
                                candidate_pid = pid
                                min_dis = dist
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

            if taskid not in cur_task_set:
                action_pid = random.randint(1, len(par_f)) # start from 1
                action_taskid = taskid
                action = ["pick-rand", action_pid, action_taskid]
                action_idlist.append(action_pid)
                actions.append(action)

        length = len(action_idlist)
        for i in range(length, len(task_f)):
            action_idlist.append(0)
        return action_idlist

    def Train_Data(self, policy, train_epoch, train_data, rescale = True, AC = True):
        #samples = self.random_sampling()
        #print [self.replay_Q[i] for i in samples]
        #print "sample ok"
        #print len(self.replay_obs)
        #print len(self.replay_rew)
        #print len(self.replay_next)
        #print self.replay_Q
        samples_obs, samples_act, samples_epr, samples_val, samples_next = train_data
        self.sum_step += 1
        if AC:
            self.base_v = samples_val
        else:
            mean_reward = np.mean(samples_epr)
            self.base_v = 0.1 * mean_reward + 0.9 * self.base_v
        #self.base_v = (self.base_v * (self.sum_step - 1) + mean_reward) / self.sum_step 
        print "base_v:"
        print self.base_v
        if rescale: 
            origin_epr = samples_epr
            #for i in range(len(samples_epr)):
            #    samples_epr[i] = 1.0
            #    if self.task_mask == 1:
            #        samples_act[i] = self.Greedy_action_mask(samples_obs[i])
            #    else:
            #        samples_act[i] = self.Greedy_action(samples_obs[i])
            #max_id = np.argmax(samples_epr)
            #min_id = np.argmin(samples_epr)
            #for i in range(len(samples_epr)):
            #    samples_epr[i] = 0.0
            #samples_epr[max_id] = 1.0
            #samples_epr[min_id] = -1.0
            #for i in range(len(samples_act)):
            #    for j in range(len(samples_act[i])):
            #        samples_act[i][j] = 5
            #samples_epr[0] = 1.0
            max_epr = 0
            mean_epr = 0
            for i in range(len(samples_epr)):
                samples_epr[i] -= self.base_v[i]
                mean_epr += samples_epr[i] * samples_epr[i]
                if max_epr < samples_epr[i] * samples_epr[i]:
                    max_epr = samples_epr[i] * samples_epr[i]
            mean_epr /= len(samples_epr)
            greedy_count = 0
            great = 0.001
            no_greedy_count = 0
            no_great = 0.001
            greedy_percent = 0.0
            no_greedy_percent = 0.0
            for i in range(len(samples_epr)):
                if self.task_mask == 1:
                    greedy_act = self.Greedy_action_mask(samples_obs[i])
                else:
                    greedy_act = self.Greedy_action(samples_obs[i])
                if samples_epr[i] > 0 and greedy_act == samples_act[i]:
                    greedy_count += 1
                    greedy_percent += samples_epr[i]
                    great += 1
                elif samples_epr[i] > 0:
                    #samples_epr[i] = 0.0
                    great += 1
                    no_greedy_percent += samples_epr[i]
                elif samples_epr[i] < 0 and greedy_act == samples_act[i]:
                    #samples_epr[i] = 0.0
                    no_greedy_count += 1
                    greedy_percent += samples_epr[i]
                    no_great += 1
                else:
                    no_great += 1
                    no_greedy_percent += samples_epr[i]
            print "great_occupy:" + str(float(greedy_count) / float(great)) + "_" + str(float(no_greedy_count) / float(no_great))
            print "greedy_percent:" + str(greedy_percent) + "_" + str(no_greedy_percent)
            print "max_epr:" + str(max_epr) + "mean_epr:" + str(mean_epr)
            print "samples_epr:"
            print samples_epr
            print "greedy action:"
            print samples_act
        self.Train_Critic(train_data, train_epoch)
        policy.model.train_model(samples_obs, samples_act, samples_epr, origin_epr, samples_next, train_epoch)
    
    def Train_Critic(self, train_data, train_epoch):
        print "Train Critic"
        samples_obs, samples_act, samples_epr, samples_val, samples_next = train_data
        #for i in range(len(samples_epr)):
        #    samples_epr[i] -= samples_val[i]
        samples_epr = [1.0 for i in range(len(samples_epr))]
        for i in range(100):
            self.Critic.train_model(samples_obs, samples_act, samples_epr, samples_val, samples_next, train_epoch)
