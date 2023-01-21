import unittest
import pogym

from agents import QLearningFSCAgent
from fsc import FiniteHistoryFSC
from tests import test_qlearning


class QLearningFSCTestCase(test_qlearning.QLearningTestCase):
    k = 0

    @classmethod
    def get_qlearning_agent(cls, initial_q=1, discount=0.9, alpha=0.5, epsilon=1):
        env = pogym.make("Tiger-v0", new_step_api=True)
        env.reset(seed=123)
        fsc = FiniteHistoryFSC.make_uniform_fsc(2, 3, k=cls.k)
        agent = QLearningFSCAgent(env, fsc=fsc, seed=123, discount=discount, alpha=alpha, epsilon=epsilon)
        agent.Q_table.fill(initial_q)
        return agent


class QLearningFSCTestCase2(QLearningFSCTestCase):
    O, A, NO = 0, 1, 1
    S = 22   # encoding of the initial memory and observation 0
    k = 1


if __name__ == '__main__':
    unittest.main()
