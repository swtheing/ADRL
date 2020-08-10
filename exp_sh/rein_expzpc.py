import sys
import os

from Reinforce_Method.Random import *
from Reinforce_Method.Greedy import *
from Config.config import *
from Reinforce_Method.MC_PG import *
from Reinforce_Method.Actor_Critic import *
from Reinforce_Method.PPO import *
#from Data_Generator_mc.env_trajectory import *
from Data_Generator.env_mc import *

if __name__ == "__main__":

    GPU_ID = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    # Environment
    if sys.argv[1] == "RANDOM":
        config = Random_config()
        game = "random_trj"
        env = env_trajectory(game, config)
        rand = Random(config, game, env)
        rand.run_test(5000)
    elif sys.argv[1] == "GREEDY":
        config = Greedy_config()
        game = "greedy_trj"
        env = env_trajectory(game, config)
        greedy = Greedy(config, game, env)
        greedy.run_test(5000)
    elif sys.argv[1] == "WOFF":
        config = WorstOff_config()
        game = "warst_off"
        env = env_trajectory(game, config)
        worst_off= WorstOff(config, game, env)
        worst_off.run_test(5000)
    elif sys.argv[1] == "AC":
        config = Traffic_config()
        game = "trajectory"
        env = env_trajectory(game, config)
        AC = Actor_Critic(config, game, env)
        AC.Policy_Iteration()
    elif sys.argv[1] == "MC_PG":
        if len(sys.argv) < 5:
            print "Wrong argv num!"
            exit(0)
        max_step = int(sys.argv[3])
        task_num = int(sys.argv[4])
        par_num = int(sys.argv[5])

        # model config
        config = Traffic_config()
        config.max_step = max_step
        config.episode_task_num = task_num
        config.participant_num = par_num
        config.max_task_size = task_num
        config.max_par_size = par_num
        config.env_var = False
        config.poisson_lamda = task_num

        game = "trajectory"

        # env config
        env_config = {}
        env_config["config_path"] = "Data_Generator/Config/config.random.dat"
        env_config["max_step"] = max_step
        env_config["episode_task_num"] = task_num
        env_config["max_task_size"] = task_num
        env_config["participant_num"] = par_num
        env_config["max_par_size"] = par_num
        env_config["env_var"] = False
        env_config["poisson_lamda"] = task_num

        env = env_test(env_config)
        PG = MC_PG(config, game, env)
        PG.Policy_Iteration()
    elif sys.argv[1] == "PPO":
        config = Traffic_config()
        game = "trajectory"
        env = env_trajectory(game, config)
        PG = PPO(config, game, env)
        PG.Policy_Iteration()
