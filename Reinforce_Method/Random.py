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

    def run_test(self, config):
        total_reward = 0.0
        total_episode = 0
        for i in range(0, 20):
            # random
            for pid in self.env.simulator.participants:
                if self.env.simulator.participants[pid][1] == ParticipantState["available"]:
                    action_pid = pid
                    break
            for taskid in self.env.simulator.pending_schedules:
                action_task = taskid
                break
            action = ["pick", action_pid, action_task]
            observation, reward, done, info = self.env.step(action)
            if done:
                self.env.reset()
                total_episode += 1
                total_reward += reward
                print "episode: %d, reward: %s" % (total_episode, reward)
        ave_reward = total_reward / total_episode
        print "ave_reward: %s" % ave_reward










