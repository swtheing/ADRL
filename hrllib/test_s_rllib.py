import ray
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import numpy as np
from Data_Generator.env_toy import env_toy
from Data_Generator.env_test import env_test
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from tf_model import Dense
env_config = {}
env_config["preprocess"] = "DNN"
env_config["config_path"] = "Data_Generator/Config/config.random.dat"

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 2
config["eager"] = False
config["env_config"] = env_config

class OneHotPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (self._obs_space.n, )

    def transform(self, observation):
        self.check_shape(observation)
        arr = np.zeros(self._obs_space.n)
        arr[observation] = 1
        return arr

    def write(self, observation, array, offset):
        array[offset + observation] = 1

# Can optionally call trainer.restore(path) to load a checkpoint.
class MyPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (4, 4, 1)

    def transform(self, observation):
        arr = np.zeros(16, )
        arr[observation] = 1
        return arr.reshape(4, 4, 1)

#ModelCatalog.register_custom_preprocessor("my_prep", OneHotPreprocessor)
#config["model"]["custom_preprocessor"] = "my_prep" 

ModelCatalog.register_custom_model("my_model", Dense)
config["model"]["custom_model"] = "my_model"

trainer = ppo.PPOTrainer(config=config, env=env_toy)

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
