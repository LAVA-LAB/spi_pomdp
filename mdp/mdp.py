import tqdm
import numpy as np
import numpy.ma as ma

from fsc import FSC
from utils import EncoderDecoder


class MDP:
    def __init__(
        self,
        ns: int,
        na: int,
        discount: float,
        transition: np.array,
        reward: np.array,
        initial_state_distribution: np.array,
        terminate_prob: np.array,
        encoder: EncoderDecoder,
        fsc: FSC = None,
        sa_counter: np.array = None,
        n_wedge: int = 0
    ):
        self.ns, self.na = ns, na
        self.discount = discount
        self.transition = transition
        self.reward = reward
        self.initial_state_distribution = initial_state_distribution
        self.non_terminate = 1 - terminate_prob
        self.encoder = encoder
        self.fsc = fsc
        self.sa_counter = sa_counter
        self.n_wedge = n_wedge
        self.mask = self.compute_mask()

        self.value_update = self.value_update_basic
        self.get_greedy_policy = self.get_greedy_policy_basic

    def solve(self, residual_tolerance=0.1, verbose=False) -> FSC:
        v_ = np.zeros(self.ns)
        residual = 1
        with tqdm.trange(1, 0, 1, disable=not verbose, desc=f"solving mdp with {self.ns} states") as pbar:
            while residual > residual_tolerance:
                v = v_.copy()
                residual = 0
                for s in range(self.ns):
                    v_[s] = self.value_update(v, s)
                    residual = max(residual, abs(v_[s] - v[s]))
                pbar.set_postfix(dict(residual=residual))
                pbar.update(1)
        return self.compute_fsc(v_)

    def value_update_basic(self, v, s):
        return max(
            self.q_values(v, s)
        )

    def q_values(self, v, s):
        return np.array([
            self.reward[s, a] +
            self.non_terminate[s, a] * self.discount * sum(p * v[s_] for s_, p in self.successors(s, a))
            for a in range(self.na)
        ])

    def successors(self, s, a):
        state_action_transition = self.transition[s, a]
        for ind in np.nonzero(state_action_transition)[0]:
            yield ind, state_action_transition[ind]

    def compute_fsc(self, v_):
        policy = self.get_greedy_policy(v_)
        new_fsc = self.fsc.copy()
        new_action_dist = self.get_fsc_action_distribution(policy)
        new_fsc.set_action_distribution(new_action_dist)
        return new_fsc

    def get_greedy_policy_basic(self, v_):
        policy = np.zeros((self.ns, self.na))
        for s in range(self.ns):
            q = self.q_values(v_, s)
            policy[s, np.argmax(q)] = 1.0
        return policy

    def get_fsc_action_distribution(self, policy):
        new_action_dist = self.fsc.action_distribution.copy()
        new_action_dist.fill(0)
        for s in range(self.ns):
            m, o = self.encoder.decode(s)
            for a in range(self.na):
                new_action_dist[m, o, a] = policy[s, a]
        return new_action_dist

    def compute_mask(self):
        return self.sa_counter >= self.n_wedge

    def set_n_wedge_and_update_mask(self, n_wedge):
        self.n_wedge = n_wedge
        self.mask = self.compute_mask()

    def spibb(self) -> 'MDP':
        self.value_update = self.value_update_spibb
        self.get_greedy_policy = self.get_greedy_policy_spibb
        return self

    def value_update_spibb(self, v, s):
        q_values = self.q_values(v, s)
        new_pi = self.get_bootstrapping_policy(q_values, s)
        return q_values @ new_pi

    def get_greedy_policy_spibb(self, v):
        policy = np.zeros((self.ns, self.na))
        for s in range(self.ns):
            q_values = self.q_values(v, s)
            policy[s] = self.get_bootstrapping_policy(q_values, s)
        return policy

    def get_bootstrapping_policy(self, q_values, s):
        m, o = self.encoder.decode(s)
        pib = self.fsc.action_distribution[m, o]
        p = pib[self.mask[s]].sum()
        new_pi = pib.copy()
        if p > 0:
            new_pi[self.mask[s]] = 0
            greedy_action = np.argmax(ma.masked_array(q_values, mask=np.bitwise_not(self.mask[s])))
            new_pi[greedy_action] = p
        new_pi /= np.sum(new_pi)
        return new_pi

    def basic(self) -> 'MDP':
        self.value_update = self.value_update_basic
        self.get_greedy_policy = self.get_greedy_policy_basic
        return self


def estimate_mdp(
    dataset: list,
    ns: int,
    na: int,
    discount, fsc: FSC = None,
    n_wedge: int = 0
) -> 'MDP':
    if fsc is None:
        fsc = FSC.make_uniform_fsc(ns, na, 1)

    encoder = EncoderDecoder([range(fsc.nM), range(ns)])
    ns = encoder.size

    transition = np.zeros((ns, na, ns))
    reward = np.zeros((ns, na))
    probability_terminate = np.zeros((ns, na))

    counter_t = np.zeros(transition.shape)
    acc_reward = np.zeros(reward.shape)
    termination_counter = np.zeros(reward.shape)
    initial_state_counter = np.zeros(ns)
    for episode in dataset:
        fsc.reset()
        for t, (o, a, r, next_o, terminated) in enumerate(episode):
            s = encoder.encode(fsc.current_memory, o)
            fsc.update_memory(o, a)
            next_s = encoder.encode(fsc.current_memory, next_o)
            if t == 0:
                initial_state_counter[s] += 1
            counter_t[s, a, next_s] += 1
            acc_reward[s, a] += r
            termination_counter[s, a] += int(terminated)
    sa_counter = np.sum(counter_t, 2)

    np.divide(counter_t, sa_counter[:, :, np.newaxis], out=transition, where=sa_counter[:, :, np.newaxis] > 0)
    np.divide(acc_reward, sa_counter, out=reward, where=sa_counter > 0)
    np.divide(termination_counter, sa_counter, probability_terminate, where=sa_counter > 2)

    isd = initial_state_counter / len(dataset)

    return MDP(ns, na, discount, transition, reward, isd, probability_terminate, encoder, fsc, sa_counter, n_wedge)
