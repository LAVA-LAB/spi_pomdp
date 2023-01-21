import unittest
import numpy as np
import pogym

from agents import Agent
from agents import QLearning
from agents import RandomAgent
from utils import evaluate_agent
from utils import run
from utils.training import discounted_return


class DiscountedReturnTestCase(unittest.TestCase):
    def test_empty(self):
        self.assertAlmostEqual(discounted_return(0.9, []), 0)

    def test_one_reward(self):
        self.assertAlmostEqual(discounted_return(0.9, [1]), 1)

    def test_two_rewards(self):
        self.assertAlmostEqual(discounted_return(0.9, [1] * 2), 1.9)

    def test_hundred_rewards(self):
        d, n = 0.95, 100
        self.assertAlmostEqual(discounted_return(d, [1] * n), (1 - d**n)/(1 - d))


class SeedingTestCase(unittest.TestCase):
    @staticmethod
    def run_agent_env(agent_class):
        env = pogym.make("Tiger-v0", new_step_api=True)
        env.reset(seed=123)
        agent = agent_class(env, seed=123)
        return run(agent, env, 2, 0.95, verbose=False)

    def test_random_agents_with_seed(self):
        self.assertListEqual(self.run_agent_env(RandomAgent), self.run_agent_env(RandomAgent))

    def test_qlearning_with_seed(self):
        self.assertListEqual(self.run_agent_env(QLearning), self.run_agent_env(QLearning))

    def test_evaluate_deterministic_reward(self):
        env = pogym.make("Tiger-v0", new_step_api=True)
        agent = Agent(env)
        agent.act = lambda obs, training: 2
        self.assertEqual(evaluate_agent(agent, env, 20, 1), -300)

    def test_evaluate_full_discount(self):
        env = pogym.make("Tiger-v0", new_step_api=True)
        agent = Agent(env)
        agent.act = lambda obs, training: 2
        self.assertEqual(evaluate_agent(agent, env, 20, 0), -1)

    def test_evaluate_stochastic_rewards(self):
        env = pogym.make("Tiger-v0", new_step_api=True)
        agent = Agent(env)
        agent.act = lambda obs, training: 1
        self.assertAlmostEqual(evaluate_agent(agent, env, 10000, 0.9), -45, delta=2)

    def test_evaluate_does_not_change_q_values(self):
        env = pogym.make("Tiger-v0", new_step_api=True)
        agent = QLearning(env)
        old_table = agent.Q_table.copy()
        evaluate_agent(agent, env, 2, 0.9)
        self.assertTrue(np.array(old_table == agent.Q_table).all())

    def test_training_updates_q_values(self):
        env = pogym.make("Tiger-v0", new_step_api=True)
        agent = QLearning(env)
        agent.Q_table[:, :] = 0
        old_table = agent.Q_table.copy()
        run(agent, env, 10, 0.9)
        self.assertTrue(np.array(old_table != agent.Q_table).any())


if __name__ == '__main__':
    unittest.main()
