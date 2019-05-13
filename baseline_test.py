import sys
import os
from Reinforce_Method.Random import *
from Reinforce_Method.Greedy import *
from Config.config import *
from Data_Generator.env_trajectory import *

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Environment
    if sys.argv[1] == "RANDOM":
        config = Random_config()
        game = "env_trajectory"
        env = env_trajectory(game, config)
        rand = Random(config, game, env)
        for i in range(150):
            reward, ave_pending_time, ave_fare_amount = rand.run_test(config.replay_match * config.max_step)
            print "FINAL_RES:\titer:%s\tave_reward:%s\tave_pending_time:%s\tave_fare_amount:%s" \
                % (i, reward, ave_pending_time, ave_fare_amount)
    elif sys.argv[1] == "GREEDY":
        config = Greedy_config()
        game = "env_trajectory"
        env = env_trajectory(game, config)
        greedy = Greedy(config, game, env)
        for i in range(150):
            reward, ave_pending_time, ave_fare_amount = greedy.run_test(config.replay_match * config.max_step)
            print "FINAL_RES:\titer:%s\tave_reward:%s\tave_pending_time:%s\tave_fare_amount:%s" \
                % (i, reward, ave_pending_time, ave_fare_amount)
