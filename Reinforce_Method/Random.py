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
        for i in range(1, 20):
            # random
            for pid in self.env.simulator.participants:
                if self.env.simulator.participants[pid][1] == ParticipantState["available"]:
                    action_pid = pid
                    break
            for taskid in self.env.simulator.pending_schedules:
                action_task = taskid
                break
            action = ["pick", action_pid, action_task]
            self.env.step(action, config, i)










