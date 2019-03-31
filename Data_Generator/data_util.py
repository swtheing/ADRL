

class env(object):
    def __init__(self, name):
        self.name = name

    def reset(self):
        raise NotImplementedError("Abstract Method")
    def step(self, action):
        raise NotImplementedError("Abstract Method")

