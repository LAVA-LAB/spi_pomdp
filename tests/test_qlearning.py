import unittest
import numpy as np
import pogym
from numpy import testing as np_testing

from agents import QLearning


class QLearningTestCase(unittest.TestCase):
    O, A, NO = 0, 1, 1
    S = O  # S is used for direct access to the Q-table

    @classmethod
    def get_qlearning_agent(cls, initial_q=1, discount=0.9, alpha=0.5, epsilon=1):
        env = pogym.make("Tiger-v0", new_step_api=True)
        env.reset(seed=123)
        agent = QLearning(env, seed=123, discount=discount, alpha=alpha, epsilon=epsilon)
        agent.Q_table.fill(initial_q)
        return agent

    def test_initial_step(self):
        q_agent = self.get_qlearning_agent()
        self.assertEqual(q_agent.step, 0)
        self.assertEqual(q_agent.episode, 0)

    def test_first_step(self):
        q_agent = self.get_qlearning_agent()
        q_agent.update(self.O, self.A, 0, self.NO, False, {})
        self.assertEqual(q_agent.step, 1)
        self.assertEqual(q_agent.episode, 0)

    def test_first_episode(self):
        q_agent = self.get_qlearning_agent()
        q_agent.update(self.O, self.A, 0, self.NO, False, {})
        q_agent.end_episode()
        self.assertEqual(q_agent.step, 0)
        self.assertEqual(q_agent.episode, 1)

    def test_partial_update(self):
        q_agent = self.get_qlearning_agent(alpha=0.5)
        q_agent.update(self.O, self.A, 1, self.NO, False, {})
        self.assertEqual(q_agent.Q_table[self.S, self.A], 1.45)

    def test_full_update(self):
        q_agent = self.get_qlearning_agent(alpha=1)
        q_agent.update(self.O, self.A, 1, self.NO, False, {})
        self.assertEqual(q_agent.Q_table[self.S, self.A], 1.9)

    def test_update_terminated(self):
        q_agent = self.get_qlearning_agent()
        q_agent.update(self.O, self.A, 2, self.NO, True, {})
        self.assertEqual(q_agent.Q_table[self.S, self.A], 1.5)

    def test_greedy_action(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        self.assertEqual(q_agent._greedy_action(self.S), self.A)

    def test_greedy_actions(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        q_agent.Q_table[self.S, self.A + 1] = 2
        self.assertListEqual(list(q_agent._greedy_actions(self.S)), [self.A, self.A + 1])

    def test_act_no_exploration_no_training(self):
        q_agent = self.get_qlearning_agent(epsilon=0)
        q_agent.Q_table[self.S, self.A] = 2
        self.assertEqual(q_agent.act(self.O, training=False), self.A)

    def test_act_pure_exploration_no_training(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        self.assertEqual(q_agent.act(self.O, training=False), self.A)

    def test_act_pure_exploration_training(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        n = 1000
        prob_a = float(sum(q_agent.act(self.O, training=True) == self.A for _ in range(n))) / n
        self.assertAlmostEqual(prob_a, 1./q_agent.num_actions, places=1)

    def test_act_no_exploration_training(self):
        q_agent = self.get_qlearning_agent(epsilon=0)
        q_agent.Q_table[self.S, self.A] = 2
        self.assertEqual(q_agent.act(self.O, training=True), self.A)

    def test_get_fsc(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        q_agent.Q_table[self.S, self.A + 1] = 3
        fsc = q_agent.export_fsc()
        self.assertEqual(fsc.get_action(self.O), self.A + 1)

    def test_get_fsc_softmax(self):
        q_agent = self.get_qlearning_agent()
        q_agent.Q_table[self.S, self.A] = 2
        q_agent.Q_table[self.S, self.A + 1] = 3
        fsc = q_agent.export_fsc(beta=0.2)
        np_testing.assert_allclose(
            fsc.action_distribution[fsc.nM - 1][self.O],
            np.array([0.26930749917, 0.32893292228, 0.40175957853])
        )


if __name__ == '__main__':
    unittest.main()
