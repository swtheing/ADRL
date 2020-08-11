import ray
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import numpy as np
#from Data_Generator.env_toy import env_toy
from Data_Generator.env_test import env_test
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from tf_model import Dense
import os
import sys

env_config = {}
env_config["preprocess"] = "DNN"
env_config["config_path"] = "Data_Generator/Config/config.random.dat"

GPU_ID = sys.argv[1]
max_step  = int(sys.argv[2])
episode_task_num = int(sys.argv[3])
#max_task_size   5   #int
participant_num = int(sys.argv[4])
#max_par_size    10  #int

env_config["max_step"] = max_step
env_config["episode_task_num"] = episode_task_num
env_config["max_task_size"] = episode_task_num
env_config["participant_num"] = participant_num
env_config["max_par_size"] = participant_num


ray.init()
DEFAULT_CONFIG = {
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE(lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "sample_batch_size": 200,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    
    "num_sgd_iter": 30,
    
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    "lr_schedule": None,
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    "simple_optimizer": False,
}
#config = dqn.DEFAULT_CONFIG.copy()
config = ppo.DEFAULT_CONFIG.copy()
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID #'3'
config["ignore_worker_failures"] = True
config["sgd_minibatch_size"] = episode_task_num
config["train_batch_size"] = 4000
config["lr"] = 5e-4
config["num_gpus"] = 1
config["num_workers"] = 1
config["eager"] = False
config["env_config"] = env_config

# class OneHotPreprocessor(Preprocessor):
#     def _init_shape(self, obs_space, options):
#         return (self._obs_space.n, )

#     def transform(self, observation):
#         self.check_shape(observation)
#         arr = np.zeros(self._obs_space.n)
#         arr[observation] = 1
#         return arr

#     def write(self, observation, array, offset):
#         array[offset + observation] = 1

# # Can optionally call trainer.restore(path) to load a checkpoint.
# class MyPreprocessor(Preprocessor):
#     def _init_shape(self, obs_space, options):
#         return (4, 4, 1)

#     def transform(self, observation):
#         arr = np.zeros(16, )
#         arr[observation] = 1
#         return arr.reshape(4, 4, 1)

#ModelCatalog.register_custom_preprocessor("my_prep", OneHotPreprocessor)
#config["model"]["custom_preprocessor"] = "my_prep" 

#ModelCatalog.register_custom_model("my_model", Dense)
#config["model"]["custom_model"] = "my_model"

trainer = ppo.PPOTrainer(config=config, env=env_test)
#trainer = dqn.DQNTrainer(config=config, env=env_test)

for i in range(150):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
