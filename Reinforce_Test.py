import sys
import gym
import os
from Reinforce_Method.DQN import *
from Reinforce_Method.SARSA import *
from Reinforce_Method.DirectPolicySearch import *
from Reinforce_Method.Policy_Generator import *
from Reinforce_Method.ActorCritic import *
from Reinforce_Method.MC_PG import *
from Reinforce_Method.MC_Q import *
from Reinforce_Method.DDQN import *
from Config.config import *
from Data_Generator.env_lm import *
from Data_Generator.env_atari import *
from Data_Generator.env_toy import *
from Data_Generator.env_test_continue import *
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if sys.argv[1] == "DQN_DNN":
    #DQN with DNN
        config = DQN_config()
        game = "Pong-v0" #0 stays 1 up 2 down
        #env = env_atari(game, config)
        env = gym.make(game)
        DQN = DQN(config, game, env)
        DQN.Policy_Iteration(config.inner_loop)

    elif sys.argv[1] == "DQN_DNN_TOY":
        config = DQN_Toy_config()
        game = "toy"
        env = env_toy("DNN")
        DQN = DQN(config, game, env)
        DQN.Policy_Iteration()

    elif sys.argv[1] == "DQN_CNN_TOY":
        config = DQN_Toy_Cnn_config()
        game = "toy"
        env = env_toy("CNN")
        DQN = DQN(config, game, env)
        DQN.Policy_Iteration()

    elif sys.argv[1] == "MC_Q":
        config = MC_config()
        game = "Pong-v0"
        env = gym.make(game)
        MC_Q = MC_Q(config, game, env)
        MC_Q.Policy_Iteration()

    elif sys.argv[1] == "PG_CON_TOY":
        config = PG_TOY_config()
        game = "toy"
        env = env_con_toy(game)
        PG = MC_PG(config, game, env)
        PG.Policy_Iteration()
    
    elif sys.argv[1] == "DDQN_DNN":
    #DQN with DNN
        config = DDQN_config()
        game = "Pong-v0" #0 stays 1 up 2 down
        #env = env_atari(game, config)
        env = gym.make(game)
        DDQN = DDQN(config, game, env)
        DDQN.Policy_Iteration()

    elif sys.argv[1] == "DDQN_CNN":
    #DQN with CNN
        #config = Cnn_config_Atari()
        config = Cnn_raw_config_Atari()
        game = "Pong-v0" #0 stays 1 up 2 down
        #env = env_atari(game, config)
        env = gym.make(game)
        DQN = DDQN(config, game, env)
        DQN.Policy_Iteration()

    #SARSA
    elif sys.argv[1] == "SARSA":
        config = SARSA_config()
        game = "Pong-v0"
        #env = env_atari(game, config)
        env = gym.make(game)
        SARSA = SARSA(config, game, env)
        SARSA.Policy_Iteration()

    #DPS
    elif sys.argv[1] == "DPS":
        #Directed Policy Search
        config = DPS_config()
        game = "Pong-v0"
        env = gym.make(game)
        DPS = DPS(config, game, env)
        DPS.Policy_Search()

    #AC
    elif sys.argv[1] == "AC":
        #Actor-Critic
        config = AC_config()
        game = "Pong-v0"
        #env = env_atari(game, config)
        env = gym.make(game)
        AC = ActorCritic(config, game, env)
        AC.Policy_Iteration() 
    
    #REINFORCE
    elif sys.argv[1] == "REINFORCE_DNN":
        config = PG_config()
        game = "Pong-v0"
        env = gym.make(game)
        MC_PG = MC_PG(config, game, env)
        MC_PG.Policy_Iteration()

    elif sys.argv[1] == "REINFORCE_CNN":
        config = PG_CNN_config()
        game = "Pong-v0"
        env = gym.make(game)
        MC_PG = MC_PG(config, game, env)
        MC_PG.Policy_Iteration()

    elif sys.argv[1] == "REINFORCE_LM":
        config = PG_LM_config()
        game = "layer_lm"
        env = env_lm(game, config)
        MC_PG = MC_PG(config, game, env)
        MC_PG.Policy_Iteration()
     
    #Policy Generator
    elif sys.argv[1] == "PGen": 
        config = GAN_policy_config()
        game = "Pong-v0"
        env = gym.make(game)
        GAN_policy = Policy_Generator(config, game, env)
        GAN_policy.Policy_Search()

