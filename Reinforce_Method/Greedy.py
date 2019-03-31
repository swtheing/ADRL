
from Data_Generator.env_trj import *
from Data_Generator.env_trj.state_generator import *

class Greedy():
    def __init__(self, config, game_name, env):
        #model = Perceptron(game_name, None, config, "MSE", config.pre)
        #Reinforce_Suite.__init__(self, config, model, env)
        self.env = env

    def run_test(self, config, total_step):
        total_reward = 0.0
        total_episode = 0
        observation = self.env.reset()
        for i in range(total_step):
            # greedy choose, every step chose one
            action_taskid = -1
            action_pid = -1
            cur_pid_set = set()
            cur_task_set = set()
            actions = []
            for taskid in observation.pending_schedules:
                pick_flag = 0
                if taskid in cur_task_set:
                    continue
                t_start_pos = observation.pending_schedules[taskid][3][0:2]
                action_taskid = taskid
                min_dis = -1.0
                for pid in observation.participants:
                    if (pid not in cur_pid_set) and observation.participants[pid][1] == ParticipantState["available"]:
                        p_start_pos = observation.participants[pid][3]
                        distance = observation.trajector.get_distance( \
                            t_start_pos[0], t_start_pos[1], p_start_pos[0], p_start_pos[1])
                        if distance < min_dis or min_dis < 0:
                            action_pid = pid
                            min_dis = distance
                            pick_flag = 1
                if pick_flag == 0:
                    continue
                cur_pid_set.add(action_pid)
                cur_task_set.add(action_taskid)
                action = ["pick", action_pid, action_taskid]
                print action
                actions.append(action)
            new_observation, reward, done, info = self.env.step(actions)
            observation = new_observation
            if done:
                total_episode += 1
                total_reward += reward
                observation = self.env.reset()
                print "episode: %d, reward: %s" % (total_episode, reward)
        ave_reward = total_reward / total_episode
        print "ave_reward: %s" % ave_reward


