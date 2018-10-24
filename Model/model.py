import sys
class model():
    def __init__(self, model_name, game_name, config):
        self.model_name = model_name
        self.game_name = game_name


    def init_model(self, path):
        raise NotImplementedError("Abstract Method")
    def feature_extract(self, data):
        raise NotImplementedError("Abstract Method")
    def save_model(self, path, episode_number):
        raise NotImplementedError("Abstract Method")
    def train_model(self, data, action, reward, value, epoch, direct = False):
        raise NotImplementedError("Abstract Method")
    def test_model(self, data):
        raise NotImplementedError("Abstract Method")

