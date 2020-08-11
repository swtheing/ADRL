import sys

class Traffic_config(object):
    # default
    aim_day_num = 1
    trajector_sampling_size = 100
    default_ave_speed = 0.004 # 40 / 85.176 / 3600 * 300 

    max_step = 1
    episode_task_num = 1
    participant_num = 2
    max_task_size = 1
    max_par_size = participant_num
    task_mask = 0
    task_feature_size = 15
    par_feature_size = 15
    hidden_size = 256
    dropout_prob = 0.2
    multi_task = False

    # distribution
    env_var = False
    poisson_lamda = 8
    poisson_episode_num = 40
    normal_mu = 0
    normal_sigma = 0.3
    normal_episode_num = 5

    task_data_path = "Data_Generator/env_trj/data/tasks"
    trajectory_data_path = "Data_Generator/env_trj/data/trajectory"
    log_file_path = "output/traffic.state.log.test"

    epoch = 100000
    num_block = 8
    num_heads = 8
    eval_epoch = 1
    train_epoch = 1
    explore_iter = 0
    episilon = 0.0
    delay_update = 0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    
    epsilon = 0.000015
    learning_rate = 0.1
    epp = 0.5
    batch_size = 6400
    replay_match = 6400
    replay_size = 100000
    on_policy = True
    replay_switch = True
    dis_epi = 0.00
    decay = 0.96
    min_epi = 0.1
    min_epi = 0.1
    dropout_lamda = 0.5
    value_coffe = 0.0
    entro_coffe = 1.0
    model = "TranPtr"
    save_path = "Params/PG/Trans_ptr.ckpt"

class Random_config(object):
    # default
    aim_day_num = 1
    trajector_sampling_size = 100
    default_ave_speed = 0.004 # 40 / 85.176 / 3600 * 300 

    replay_match = 1
    max_step = 5
    episode_task_num = 2
    participant_num = 5
    max_task_size = 10
    max_par_size = participant_num
    task_mask = 1
    task_feature_size = 15
    par_feature_size = 15
    dropout_lamda = 0.5

    # distribution
    env_var = False
    poisson_lamda = 4
    poisson_episode_num = 40
    normal_mu = 0
    normal_sigma = 0.3
    normal_episode_num = 0.00005

    log_file_path = "output/random.state.log"
    task_data_path = "Data_Generator/env_trj/data/tasks"
    trajectory_data_path = "Data_Generator/env_trj/data/trajectory"

class Greedy_config(object):
    # default
    aim_day_num = 1
    trajector_sampling_size = 100
    default_ave_speed = 0.004 # 40 / 85.176 / 3600 * 300 
    random_prob = 0.0

    replay_match = 320
    max_step = 1
    episode_task_num = 1
    participant_num = 2
    task_mask = 0
    max_task_size = 20
    max_par_size = participant_num
    task_feature_size = 15
    par_feature_size = 15
    dropout_lamda = 0.5

    # distribution
    env_var = False
    poisson_lamda = 4
    poisson_episode_num = 40
    normal_mu = 0
    normal_sigma = 0.3
    normal_episode_num = 5

    task_data_path = "Data_Generator/env_trj/data/tasks"
    trajectory_data_path = "Data_Generator/env_trj/data/trajectory"
    log_file_path = "output/greedy.state.log"

class GAN_policy_config(object):
    epoch = 1000000
    eval_epoch = 40
    train_epoch = 2000
    episilon = 0.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80 * 80
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015
    device = "/device:GPU:2"
    learning_rate = 0.00025
    batch_size = 20
    replay_match = 50
    dis_epi = 0.01
    pre = False
    replay_size = 10000
    sample_size = 50
    conti_act = False
    noise_size = 30

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
    epsilon = 0.000015
    learning_rate = 0.00025
    reg_lr = 0.001
    reg_epsilon = 0.000015
    batch_size = 512
    replay_match = 50
    dis_epi = 0.01
    pre = True
    replay_size = 10000
    reg_epoch = 100
    conti_act = False

class SARSA_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 100
    episilon = 0.1 #1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015# adam_epsilon
    decay = 0.96
    learning_rate = 0.00025
    batch_size = 512
    replay_match = 50
    replay_size = 100000
    dis_epi = 0.001
    min_epi = 0.1
    conti_act = False
    save_path = "Params/SARSA/preceptron.ckpt"

class PG_LM_config(object):
    epoch = 1000000
    eval_epoch = 1
    explore_iter = 0
    train_epoch = 2
    episilon = 0.0
    gamma = 0.99
    alpha = 0.1
    feature_size = 6
    hidden_size = 15
    action_size = 7
    epsilon = 0.000015
    learning_rate = 0.005
    batch_size = 10
    replay_match = 10
    observe_id = 10000
    replay_size = 100000
    on_policy = True
    replay_switch = True
    dis_epi = 0.00
    min_epi = 0.1
    model = "Gaussian"
    delay_update = 0
    cmd = "sh run_lm.sh"
    max_step = 3
    early_stop = False
    outfile = "../../sw/code/LM/INFO_State"
    restore = "../../sw/code/LM/INFO_Res"
    lm_conf = "../../sw/code/LM/config/config.py"
    save_path = "Params/PG_LM/preceptron.ckpt"

class PG_TOY_config(object):
    epoch = 1000000
    eval_epoch = 1
    explore_iter = 0
    train_epoch = 1
    episilon = 0.0
    gamma = 0.99
    alpha = 0.1
    feature_size = 10
    hidden_size = 20
    action_size = 10
    epsilon = 0.000015
    learning_rate = 0.0005
    batch_size = 1
    replay_match = 1
    observe_id = 10000
    replay_size = 100000
    on_policy = True
    replay_switch = False
    dis_epi = 0.00
    min_epi = 0.1
    model = "Gaussian"
    delay_update = 0
    save_path = "Params/PG_Con/Gaussian.ckpt"

class MC_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 500
    explore_iter = 100
    dropout_prob = 0.5
    episilon = 1.0
    delay_update = 0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 1
    replay_size = 100000
    on_policy = True
    replay_switch = True
    dis_epi = 0.0001
    decay = 0.96
    min_epi = 0.1
    model = "DNN"
    save_path = "Params/PG/preceptron.ckpt"

class PG_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 1
    explore_iter = 0
    episilon = 0.0
    delay_update = 0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 512
    replay_match = 1
    replay_size = 100000
    on_policy = True
    replay_switch = False
    dis_epi = 0.00
    decay = 0.96
    min_epi = 0.1
    min_epi = 0.1
    model = "DNN"
    save_path = "Params/PG/preceptron.ckpt"

class PG_CNN_config(object):
    epoch = 1000000
    eval_epoch = 1
    train_epoch = 1
    explore_iter = 0
    delay_update = 0
    episilon = 0.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    ob_dims = [80, 80, 1]
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 512
    replay_match = 1
    replay_size = 100000
    on_policy = True
    replay_switch = False
    dis_epi = 0.00
    min_epi = 0.1
    min_epi = 0.1
    model = "CNN"
    decay = 0.96
    save_path = "Params/PG_CNN/preceptron.ckpt"

class DDQN_config(object):
    epoch = 10000000
    explore_iter = 200
    eval_epoch = 1
    train_epoch = 20
    episilon = 1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    model = "DNN"
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    decay = 0.96
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 1
    replay_size = 100000
    dis_epi = 0.0001
    min_epi = 0.1
    delay_update = 5
    conti_act = False
    save_path = "Params/DDQN/preceptron.ckpt"

class DQN_Toy_Cnn_config(object):
    epoch = 10000000
    explore_iter = 200
    eval_epoch = 1
    train_epoch = 800
    episilon = 1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    model = "CNN"
    alpha = 0.1
    feature_size = [4, 4, 1]
    hidden_size = 50
    action_size = 4
    decay = 0.96
    epsilon = 0.000015
    learning_rate = 0.0001
    batch_size = 16
    replay_match = 20
    replay_size = 100000
    debug = True
    dis_epi = 0.001
    min_epi = 0.01
    delay_update = 0
    conti_act = False
    save_path = "Params/DQN_Toy/Cnn.ckpt"

class DQN_Toy_config(object):
    epoch = 10000000
    explore_iter = 200
    eval_epoch = 1
    train_epoch = 50
    episilon = 1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    model = "DNN"
    alpha = 0.1
    feature_size = 4 * 4
    hidden_size = 50
    action_size = 4
    decay = 0.96
    epsilon = 0.000015
    learning_rate = 0.000001
    batch_size = 16
    replay_match = 20
    replay_size = 100000
    dis_epi = 0.001
    debug = True
    min_epi = 0.01
    delay_update = 0
    conti_act = False
    save_path = "Params/DQN_Toy/preceptron_delay.ckpt"

class DQN_config(object):
    epoch = 10000000
    explore_iter = 20
    inner_loop = 10
    eval_epoch = 1
    train_epoch = 100
    episilon = 1.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    model = "DNN"
    alpha = 0.1
    debug = False
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    decay = 0.96
    epsilon = 0.0001
    learning_rate = 0.000025
    batch_size = 32
    replay_match = 10
    replay_size = 500000
    dis_epi = 0.0001
    min_epi = 0.1
    delay_update = 0
    conti_act = False
    save_path = "Params/DQN/preceptron.ckpt"

class AC_config(object):
    epoch = 1000000
    eval_epoch = 1
    explore_iter = 0
    train_epoch = 1
    episilon = 0.0
    replay_file = "obs_replay/replay"
    observe_id = 10000
    gamma = 0.99
    alpha = 0.1
    feature_size = 80*80
    hidden_size = 100
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 256
    replay_match = 1
    replay_size = 100000
    decay = 0.96
    on_policy = True
    replay_switch = False
    delay_update = 0
    save_path = "Params/AC_DNN/dnn.ckpt"
    min_epi = 0.00
    dis_epi = 0.00
    conti_act = False

class Cnn_raw_config_Atari(object):
    epoch = 10000
    eval_epoch = 1
    train_epoch = 100
    episilon = 1.0
    dis_epi = 0.0001
    replay_file = "obs_replay/replay"
    observe_id = 200
    gamma = 0.99
    decay = 0.96
    explore_iter = 0
    model = "CNN"
    alpha = 0.1
    ob_dims = [80, 80, 1]
    history_step = 4
    hidden_size = 512
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 1
    replay_size = 500000
    min_epi = 0.1
    delay_update = 5
    conti_act = False
    save_path = "Params/DQN_CNN/cnn.ckpt"

class Cnn_config_Atari(object):
    epoch = 10000
    eval_epoch = 1
    train_epoch = 200
    episilon = 1.0
    dis_epi = 0.0001
    replay_file = "obs_replay/replay"
    observe_id = 200
    gamma = 0.99
    decay = 0.96
    explore_iter = 200
    alpha = 0.1
    ob_dims = (84, 84)
    feature_size = [84, 84, 4]
    history_step = 4
    hidden_size = 512
    action_size = 3
    epsilon = 0.000015
    learning_rate = 0.00025
    batch_size = 32
    replay_match = 1
    replay_size = 100000
    min_epi = 0.1
    model = "CNN"
    delay_update = 5
    conti_act = False
    save_path = "Params/DQN_CNN/cnn.ckpt"
    act_repeat = 4
