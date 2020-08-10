import sys
import os

from Reinforce_Method.Random import *
from Reinforce_Method.Greedy_OPT import *
from Reinforce_Method.Greedy import *
from Reinforce_Method.WorstOff import *
from Config.config import *
from Reinforce_Method.MC_PG import *
from Reinforce_Method.Actor_Critic import *
from Reinforce_Method.PPO import *
from Data_Generator.env_trajectory import *

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    max_step,task_num, par_num = 0, 0, 0
    if len(sys.argv) > 3:
        max_step = int(sys.argv[2])
        task_num = int(sys.argv[3])
        par_num = int(sys.argv[4])
    # Environment
    if sys.argv[1] == "RANDOM":
        config = Random_config()
        if len(sys.argv) > 1:
            config.max_step = max_step
            config.episode_task_num = task_num
            config.participant_num = par_num
            config.max_task_size = config.episode_task_num
            config.max_par_size = config.participant_num
            config.env_var = False
            config.poisson_lamda = task_num
        print config.episode_task_num
        game = "random_trj"
        env = env_trajectory(game, config)
        rand = Random(config, game, env)
        rand.run_test(5000)
    elif sys.argv[1] == "GREEDY":
        config = Greedy_config()
        if len(sys.argv) > 1:
            config.max_step = max_step
            config.episode_task_num = task_num
            config.participant_num = par_num
            config.max_task_size = config.episode_task_num
            config.max_par_size = config.participant_num
            config.env_var = False
            config.poisson_lamda = task_num
        game = "greedy_trj"
        env = env_trajectory(game, config)
        greedy = Greedy(config, game, env)
        greedy.run_test(500)
    elif sys.argv[1] == "WOFF":
        config = Worst_off_config()
        if len(sys.argv) > 1:
            config.max_step = max_step
            config.episode_task_num = task_num
            config.participant_num = par_num
            config.max_task_size = config.episode_task_num
            config.max_par_size = config.participant_num
            config.env_var = False
            config.poisson_lamda = task_num
        game = "worst_off"
        env = env_trajectory(game, config)
        worst_off= WorstOff(config, game, env)
        worst_off.run_test(500)
    elif sys.argv[1] == "GO":
        config = Greedy_opt_config()
        if len(sys.argv) > 1:
            config.max_step = max_step
            config.episode_task_num = task_num
            config.participant_num = par_num
            config.max_task_size = config.episode_task_num
            config.max_par_size = config.participant_num
            config.env_var = False
            config.poisson_lamda = task_num
        game = "greedy_opt"
        env = env_trajectory(game, config)
        greedy_opt= Greedy_OPT(config, game, env)
        greedy_opt.run_test(500)
    elif sys.argv[1] == "AC":
        config = Traffic_config()
        game = "trajectory"
        env = env_trajectory(game, config)
        AC = Actor_Critic(config, game, env)
        AC.Policy_Iteration()
    elif sys.argv[1] == "MC_PG":
        config = Traffic_config()
        game = "trajectory"
        env = env_trajectory(game, config)
        PG = MC_PG(config, game, env)
        PG.Policy_Iteration()
    elif sys.argv[1] == "PPO":
        config = Traffic_config()
        game = "trajectory"
        env = env_trajectory(game, config)
        PG = PPO(config, game, env)
        PG.Policy_Iteration()
