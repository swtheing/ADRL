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

    def run_test(self, config, total_step):
        total_reward = 0.0
        total_episode = 0
        observation = self.env.reset()
        for i in range(total_step):
            # random
            for pid in observation.participants:
                if observation.participants[pid][1] == ParticipantState["available"]:
                    action_pid = pid
                    break
            for taskid in observation.pending_schedules:
                action_task = taskid
                break
            action = ["pick", action_pid, action_task]
            new_observation, reward, done, info = self.env.step(action)
            observation = new_observation
            if done:
                total_episode += 1
                total_reward += reward
                observation = self.env.reset()
                print "episode: %d, reward: %s" % (total_episode, reward)
        ave_reward = total_reward / total_episode
        print "ave_reward: %s" % ave_reward

