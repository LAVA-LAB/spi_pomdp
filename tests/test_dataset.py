import unittest
import pogym

from agents import Agent
from utils import new_dataset
from utils.dataset import get_new_episode


# use an environment with non-uniform initial state distribution to break symmetries
TL, TR = 0.6, 0.4
NL, NR = TL * 0.85 + TR * 0.15, TL * 0.15 + TR * 0.85


def make_agent_env():
    env = pogym.make("Tiger-v0", isd=[TL, TR])
    env.reset()
    agent = Agent(env)
    agent.act = lambda obs, training: 1
    return agent, env


class DataCollectionTestCase(unittest.TestCase):
    def test_get_new_episode(self):
        agent, env = make_agent_env()
        [(observation, action, reward, next_observation, terminated)] = get_new_episode(agent, env)
        self.check_transition(observation, action, reward, next_observation, terminated)

    def test_dataset_size(self):
        agent, env = make_agent_env()
        dataset = new_dataset(agent, env, 2)
        self.assertEqual(len(dataset), 2)

    def test_dataset_first_transition(self):
        agent, env = make_agent_env()
        dataset = new_dataset(agent, env, 10)
        for episode in dataset:
            self.check_transition(*episode[0])

    def check_transition(self, observation, action, reward, next_observation, terminated):
        self.assertIn(observation, [0, 1])
        self.assertEqual(action, 1)
        self.assertTrue(terminated)
        self.assertIn(next_observation, [0, 1])
        if next_observation == 1:
            self.assertEqual(reward, -100)
        else:
            self.assertEqual(reward, 10)


if __name__ == '__main__':
    unittest.main()
