import sys
import gym
from DQN import *
from DirectPolicySearch import *
from env_atari import *
from config import *
if __name__ == "__main__":
    #DQN
    config = P_config()
    # #game = "Pong-v0"  #0 stays 1 up 2 down
    game = "Pong-v0"
    # #env = env_atari(game, config)
    env = gym.make(game)
    DQN = DQN(config, game, env)
    DQN.Policy_Iteration()

    #Directed Policy Search
    #config = DPS_config()
    #game = "Pong-v0"
    #env = gym.make(game)
    #DPS = DPS(config, game, env)
    #DPS.Policy_Search()

