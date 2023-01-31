import unittest
from gym.spaces import Discrete
import pogym


from fsc import FiniteObservationHistoryFSC
from fsc import FSCWrapper


def make_fsc_env(k):
    core_env = pogym.make("Tiger-v0")
    fsc = FiniteObservationHistoryFSC.make_uniform_fsc(2, 3, k)
    env = FSCWrapper(core_env, fsc=fsc)
    env.reset(seed=123)
    return env, core_env


class FSCWrapperTestCase(unittest.TestCase):
    def test_reset_0(self):
        env, _ = make_fsc_env(0)
        self.assertEqual(env.reset(seed=123), env.reset(seed=123))

    def test_reset_2(self):
        env, _ = make_fsc_env(2)
        self.assertEqual(env.reset(seed=123), env.reset(seed=123))

    def test_observation0(self):
        env, _ = make_fsc_env(0)
        self.assertEqual(env.observation_space, Discrete(2))

    def test_initial_observation(self):
        env, _ = make_fsc_env(1)
        self.assertEqual(env.observation_space, Discrete(6))
        obs, info = env.reset(seed=None)
        self.assertIn(obs, [4, 5])

    def test_step(self):
        env, _ = make_fsc_env(1)
        next_observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        if next_observation == 5:
            self.assertEqual(reward, 10)
        else:
            self.assertEqual(reward, -100)
        self.assertEqual(info, {})


if __name__ == '__main__':
    unittest.main()
