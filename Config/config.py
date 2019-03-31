import sys
class DPS_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 100
    episilon = 0.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    decay = 0.96
    learning_rate = 0.00025
    batch_size = 512
    replay_match = 50
    dis_epi = 0.01
    pre = True


class Random_config(object):
    # default
    aim_day_num = 1
    trajector_sampling_size = 100
    default_ave_speed = 0.04 # 40 / 85.176 / 3600 * 300 
    max_step = 10
    episode_task_num = 40
    participant_num = 20

    # distribution
    env_var = True
    poisson_lamda = 6
    poisson_episode_num = 40
    normal_mu = 0
    normal_sigma = 0.1
    normal_episode_num = 5

    log_file_path = "output/random.state.log"
    task_data_path = "Data_Generator/env_trj/data/tasks"
    trajectory_data_path = "Data_Generator/env_trj/data/trajectory"
    

class Greedy_config(object):
    # default
    aim_day_num = 1
    max_step = 10
    trajector_sampling_size = 100
    default_ave_speed = 0.04 # 40 / 85.176 / 3600 * 300 
    episode_task_num = 40
    participant_num = 20

    # prob
    env_var = True
    poisson_lamda = 6
    poisson_episode_num = 40
    normal_mu = 0
    normal_sigma = 0.1
    normal_episode_num = 5

    task_data_path = "Data_Generator/env_trj/data/tasks"
    trajectory_data_path = "Data_Generator/env_trj/data/trajectory"
    log_file_path = "output/greedy.state.log"


class P_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 100
    episilon = 1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    decay = 0.96
    learning_rate = 0.00025
    batch_size = 512
    replay_match = 200
    replay_size = 100000
    dis_epi = 0.001
    pre = False

class Cnn_raw_config_Atari(object):
    epoch = 10000
    eval_epoch = 1
    train_epoch = 100
    episilon = 1.0
    dis_epi = 0.001
    replay_file = "obs_replay/replay"
    observe_id = 200
    gamma = 0.99
    alpha = 0.1
    ob_dims = [210, 160, 3]
    history_step = 4
    hidden_size = 512
    action_size = 3
    decay = 0.96
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 20
    replay_size = 5000

class Cnn_config_Atari(object):
    epoch = 10000
    eval_epoch = 1
    train_epoch = 100
    episilon = 1.0
    dis_epi = 0.001
    replay_file = "obs_replay/replay"
    observe_id = 200
    gamma = 0.99
    alpha = 0.1
    ob_dims = [80, 80, 4]
    history_step = 4
    hidden_size = 512
    action_size = 3
    decay = 0.96
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 20
    replay_size = 500000
