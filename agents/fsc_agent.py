from .base import Agent
from fsc import FSC
from gym import Env


class FSCAgent(Agent):
    """
    this class acts as a wrapper around an FSC
    """
    def __init__(self, env: Env, fsc: FSC, seed=None):
        self.fsc = fsc
        super().__init__(env=env, seed=seed)

    def act(self, obs, training):
        return self.fsc.get_action(obs, step=True)

    def end_episode(self):
        self.fsc.reset()

    def _seed(self, seed):
        super()._seed(seed)
        self.fsc.seed(seed)
