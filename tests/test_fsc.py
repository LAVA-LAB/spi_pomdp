import unittest
import numpy as np
from numpy import testing as np_testing
import pogym

from agents import FSCAgent
from fsc import FSC
from fsc import FiniteHistoryFSC
from fsc import FiniteObservationHistoryFSC


class TestMemoryLessFiniteStateController(unittest.TestCase):
    def test_reset(self):
        p = FSC.make_uniform_fsc(num_observations=1, num_actions=1, memory_size=1)
        self.assertEqual(p.reset(), 0)

    def test_get_action(self):
        p = FSC.make_uniform_fsc(num_observations=1, num_actions=1, memory_size=1)
        self.assertEqual(p.get_action(0), 0)

    def test_step(self):
        p = FSC.make_uniform_fsc(num_observations=1, num_actions=1, memory_size=1)
        self.assertEqual(p._step(0, 0), 0)


class TestSimpleFiniteStateController(unittest.TestCase):
    @staticmethod
    def make_simple_fsc():
        return FSC(
            memory_transition=np.array([[[[1, 0], [0, 1]]], [[[1, 0], [0, 1]]]]),
            initial_memory_distribution=np.array([0, 1]),
            action_distribution=np.array([[[1], [1]], [[1], [1]]]))

    def test_initial_memory(self):
        p = self.make_simple_fsc()
        p.reset()
        self.assertEqual(p.current_memory, 1)

    def test_reset(self):
        p = self.make_simple_fsc()
        p.get_action(0)
        memory_node = p.reset()
        self.assertEqual(memory_node, 1)

    def test_get_action(self):
        p = self.make_simple_fsc()
        self.assertEqual(p.get_action(0), 0)

    def test_get_action_without_updating_memory(self):
        p = self.make_simple_fsc()
        p.get_action(0, step=False)
        self.assertEqual(p.current_memory, 1)

    def test_memory_update0(self):
        p = self.make_simple_fsc()
        p.get_action(0)
        self.assertEqual(p.current_memory, 0)

    def test_memory_update1(self):
        p = self.make_simple_fsc()
        p.get_action(1)
        self.assertEqual(p.current_memory, 1)

    def test_step0(self):
        p = self.make_simple_fsc()
        self.assertEqual(p._step(0, 0), 0)

    def test_step1(self):
        p = self.make_simple_fsc()
        self.assertEqual(p._step(1, 0), 1)


def make_tiger_fsc(k=1, fsc_class=FiniteHistoryFSC):
    env = pogym.make("Tiger-v0")
    fsc = fsc_class.make_uniform_fsc(2, 3, k=k)
    env.reset()
    fsc.reset()
    return fsc, env


class TestTiger1k(unittest.TestCase):

    def test_initial_memory(self):
        fsc, env = make_tiger_fsc()
        self.assertListEqual(fsc.decode(fsc.current_memory), [2, 3])

    def test_memory_is_updated(self):
        fsc, env = make_tiger_fsc()
        terminated = False
        obs, info = env.reset()
        while not terminated:
            action = fsc.get_action(obs)
            self.assertListEqual(fsc.decode(fsc.current_memory), [obs, action])
            obs, _, terminated, _, _ = env.step(action)


class TestTiger2k(unittest.TestCase):

    def test_initial_memory(self):
        fsc, env = make_tiger_fsc(2)
        self.assertListEqual(fsc.decode(fsc.current_memory), [2, 3, 2, 3])

    def test_memory_is_updated(self):
        fsc, env = make_tiger_fsc(2)
        terminated = False
        obs, info = env.reset()
        prev_obs, prev_action = 2, 3
        while not terminated:
            action = fsc.get_action(obs)
            self.assertListEqual(fsc.decode(fsc.current_memory), [prev_obs, prev_action, obs, action])
            prev_obs, prev_action = obs, action
            obs, _, terminated, _, _ = env.step(action)


class TestTiger3k(unittest.TestCase):
    k = 3

    def test_initial_memory(self):
        fsc, env = make_tiger_fsc(self.k)
        self.assertListEqual(fsc.decode(fsc.current_memory), [2, 3] * self.k)

    def test_first_ten_steps(self):
        fsc, env = make_tiger_fsc(self.k)
        terminated = False
        obs, info = env.reset()
        h = []
        t = 0
        while not terminated and t < self.k + 1:
            self.assertListEqual(fsc.decode(fsc.current_memory), [2, 3] * (self.k - t) + h)
            action = fsc.get_action(obs)
            h = h + [obs, action]
            obs, _, terminated, _, _ = env.step(action)
            t += 1


class TestTigerObservation1k(unittest.TestCase):
    def test_initial_memory(self):
        fsc, env = make_tiger_fsc(1, FiniteObservationHistoryFSC)
        self.assertListEqual(fsc.decode(fsc.current_memory), [2])

    def test_memory_is_updated(self):
        fsc, env = make_tiger_fsc(1, FiniteObservationHistoryFSC)
        terminated = False
        obs, info = env.reset()
        while not terminated:
            action = fsc.get_action(obs)
            self.assertListEqual(fsc.decode(fsc.current_memory), [obs])
            obs, _, terminated, _, _ = env.step(action)


class TestTigerObservation2k(unittest.TestCase):

    def test_initial_memory(self):
        fsc, env = make_tiger_fsc(2, FiniteObservationHistoryFSC)
        self.assertListEqual(fsc.decode(fsc.current_memory), [2, 2])

    def test_memory_is_updated(self):
        fsc, env = make_tiger_fsc(2, FiniteObservationHistoryFSC)
        terminated = False
        obs, info = env.reset()
        prev_obs, prev_action = 2, 3
        while not terminated:
            action = fsc.get_action(obs)
            self.assertListEqual(fsc.decode(fsc.current_memory), [prev_obs, obs])
            prev_obs, prev_action = obs, action
            obs, _, terminated, _, _ = env.step(action)


class TestTigerObservation6k(unittest.TestCase):
    k = 6

    def test_initial_memory(self):
        fsc, env = make_tiger_fsc(self.k, FiniteObservationHistoryFSC)
        self.assertListEqual(fsc.decode(fsc.current_memory), [2] * self.k)

    def test_first_ten_steps(self):
        fsc, env = make_tiger_fsc(self.k, FiniteObservationHistoryFSC)
        terminated = False
        obs, info = env.reset()
        h = []
        t = 0
        while not terminated and t < self.k + 1:
            self.assertListEqual(fsc.decode(fsc.current_memory), [2] * (self.k - t) + h)
            action = fsc.get_action(obs)
            h = h + [obs]
            obs, _, terminated, _, _ = env.step(action)
            t += 1


class TestFSCAgentOnTigerObservation8k(TestTigerObservation6k):
    def test_first_ten_steps(self):
        fsc, env = make_tiger_fsc(self.k, FiniteObservationHistoryFSC)
        fsc_agent = FSCAgent(env, fsc, seed=123)
        terminated = False
        obs, info = env.reset()
        h = []
        t = 0
        while not terminated and t < self.k + 1:
            self.assertListEqual(fsc.decode(fsc.current_memory), [2] * (self.k - t) + h)
            action = fsc_agent.act(obs, training=False)
            h = h + [obs]
            new_obs, reward, terminated, truncated, info = env.step(action)
            fsc_agent.update(obs, action, reward, new_obs, terminated, info)
            obs = new_obs
            t += 1


class TestMemoryExpansion(unittest.TestCase):
    fsc_class = FiniteHistoryFSC

    def test_memory_increases(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        new_fsc = fsc.expand_memory(1)
        self.assertEqual(fsc.nM, 1)
        self.assertEqual(new_fsc.nM, 12)

    def test_initial_memory_new_fsc(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        new_fsc = fsc.expand_memory(3)
        self.assertListEqual(new_fsc.decode(new_fsc.current_memory), [2, 3] * 3)

    def test_action_map_new_fsc(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        action_dist = np.array([[0, 0.1, 0.9], [0, 0.1, 0.9]])
        fsc.action_distribution[0] = action_dist
        new_fsc = fsc.expand_memory(2)
        for m in range(new_fsc.nM):
            np_testing.assert_equal(new_fsc.action_distribution[m], action_dist)

    def test_get_history(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        self.assertEqual(fsc.get_history([0, 1]), [])

    def test_get_history1(self):
        fsc, _ = make_tiger_fsc(1, self.fsc_class)
        self.assertEqual(fsc.get_history([0, 1]), [0, 1])

    def test_get_history1large(self):
        fsc, _ = make_tiger_fsc(1, self.fsc_class)
        self.assertEqual(fsc.get_history([0, 1, 1, 2]), [1, 2])


class TestMemoryExpansionFiniteObservationHistoryFSC(unittest.TestCase):
    fsc_class = FiniteObservationHistoryFSC

    def test_memory_increases(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        new_fsc = fsc.expand_memory(1)
        self.assertEqual(fsc.nM, 1)
        self.assertEqual(new_fsc.nM, 3)

    def test_initial_memory_new_fsc(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        new_fsc = fsc.expand_memory(8)
        self.assertListEqual(new_fsc.decode(new_fsc.current_memory), [2] * 8)

    def test_get_history1(self):
        fsc, _ = make_tiger_fsc(1, self.fsc_class)
        self.assertEqual(fsc.get_history([0]), [0])

    def test_get_history1large(self):
        fsc, _ = make_tiger_fsc(1, self.fsc_class)
        self.assertEqual(fsc.get_history([0, 1]), [1])

    def test_get_history_empty(self):
        fsc, _ = make_tiger_fsc(0, self.fsc_class)
        self.assertEqual(fsc.get_history([0]), [])
