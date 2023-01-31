import unittest
from mdp import estimate_mdp
from mdp import MDP

from utils import new_dataset
from .test_dataset import make_agent_env, NL, NR, TL, TR


class MDPTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        agent, env = make_agent_env()
        dataset = new_dataset(agent, env, 100000)
        cls._mdp: MDP = estimate_mdp(dataset, env.observation_space.n, env.action_space.n, 0.9)

    def test_initial_state_distribution(self):
        isd = self._mdp.initial_state_distribution
        self.assertTupleEqual(isd.shape, (2,))
        self.assertAlmostEqual(isd[0], NL, delta=0.02)
        self.assertAlmostEqual(isd[1], NR, delta=0.02)

    def test_terminate_prob(self):
        p = 1 - self._mdp.non_terminate
        self.assertTupleEqual(p.shape, (2, 3))
        self.assertAlmostEqual(p[0, 0], 0)
        self.assertAlmostEqual(p[0, 1], 1)
        self.assertAlmostEqual(p[0, 2], 0)
        self.assertAlmostEqual(p[1, 0], 0)
        self.assertAlmostEqual(p[1, 1], 1)
        self.assertAlmostEqual(p[1, 2], 0)

    def test_reward(self):
        r = self._mdp.reward
        self.assertTupleEqual(r.shape, (2, 3))
        self.assertAlmostEqual(r[0, 0], 0)
        self.assertAlmostEqual(r[0, 1], 10 * 0.85 * TL / NL - 100 * 0.15 * TR / NL, delta=2)
        self.assertAlmostEqual(r[0, 2], 0)
        self.assertAlmostEqual(r[1, 0], 0)
        self.assertAlmostEqual(r[1, 1], 10 * 0.15 * TL / NR - 100 * 0.85 * TR / NR, delta=2)
        self.assertAlmostEqual(r[1, 2], 0)

    def test_transition(self):
        t = self._mdp.transition
        self.assertTupleEqual(t.shape, (2, 3, 2))
        self.assertAlmostEqual(t[0, 1, 0], TL, delta=1)
        self.assertAlmostEqual(t[0, 1, 1], TR, delta=1)
        self.assertAlmostEqual(t[1, 1, 0], TL, delta=1)
        self.assertAlmostEqual(t[1, 1, 1], TR, delta=1)
