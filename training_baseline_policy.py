import argparse
import pickle
import os

import matplotlib.pyplot as plt
import pandas
import seaborn
import yaml

import pogym
from agents import FSCAgent
from agents import QLearningFSCAgent
from fsc import FiniteObservationHistoryFSC
from utils import evaluate_agent
from utils.training import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        default="Tiger-v0",
        type=str,
        choices=pogym.env_ids,
        help="gym env id (default=Tiger-v0)"
    )
    parser.add_argument(
        "--training_episodes",
        default=5000,
        type=int,
        help="number of training episodes (default=5000)"
    )
    parser.add_argument(
        "--evaluation_episodes",
        default=10000,
        type=int,
        help="number of evaluation episodes (default=10000)"
    )
    parser.add_argument(
        "--decaying_rate_qlearning",
        default=0.002,
        type=float,
        help="q-learning exponential decay rate (default=0.002)"
    )
    parser.add_argument(
        "--beta",
        default=0.002,
        type=float,
        help="softmax temperature to generate the behavior policy (default=0.002)"
    )
    parser.add_argument(
        "--k",
        default=1,
        type=int,
        help="size of history the FSC is tracking (default=1)"
    )
    parser.add_argument(
        "--ma_size",
        default=0.05,
        type=float,
        help="size of window for moving average -- percentage of the number of training episodes, (default=0.05)"
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="seed (default=123)"
    )
    parser.add_argument(
        "--discount",
        default=0.95,
        type=float,
        help="discount factor -- gamma (default=0.95)"
    )
    parser.add_argument(
        "--out_dir",
        default="data",
        type=str,
        help="output directory (default=data)"
    )
    return parser.parse_args()


def train_agent_on_finite_history(seed, training_episodes, env_id, decaying_rate_qlearning, beta, discount, ma_size,
                                  evaluation_episodes, k, out_dir):
    assert 0 < ma_size < 1
    window = int(training_episodes * ma_size)  # size of the moving average window
    training_episodes += window
    agent_label = f"Q-Learning FSC({k})"
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, f"{env_id}_k-{k}")

    env = pogym.make(env_id, new_step_api=True)
    fsc = FiniteObservationHistoryFSC.make_uniform_fsc(env.observation_space.n, env.action_space.n, k=k)
    env.reset(seed=seed)
    agent = QLearningFSCAgent(env, seed=seed, fsc=fsc, alpha=1, discount=discount, epsilon=0.5,
                              decaying_rate=decaying_rate_qlearning)
    episodic_returns = run(agent, env, training_episodes, discount, verbose=True, label=agent_label)

    final_agent = FSCAgent(env, agent.export_fsc(), seed)
    final_performance = evaluate_agent(final_agent, env, num_episodes=evaluation_episodes, disc=discount, verbose=True)
    baseline_fsc = agent.export_fsc(beta)
    baseline_agent = FSCAgent(env, baseline_fsc, seed)
    baseline_performance = evaluate_agent(baseline_agent, env, num_episodes=evaluation_episodes, disc=discount, verbose=True)

    with open(f'{output_prefix}.pkl', 'wb') as f:
        pickle.dump({
            "baseline_fsc": baseline_fsc,
            "baseline_performance": baseline_performance,
            "final_agent": final_agent,
            "final_performance": final_performance,
            "training_episodes": training_episodes,
            "env_id": env_id,
            "k": k,
            "decaying_rate_qlearning": decaying_rate_qlearning,
            "seed": seed,
            "beta": beta,
            "discount": discount,
        }, f)

    with open(f'{output_prefix}.yaml', 'w') as f:
        yaml.dump({
            "baseline_fsc": str(baseline_fsc),
            "baseline_performance": baseline_performance,
            "final_agent": str(final_agent),
            "final_performance": final_performance,
            "training_episodes": training_episodes,
            "env_id": env_id,
            "k": k,
            "decaying_rate_qlearning": decaying_rate_qlearning,
            "seed": seed,
            "beta": beta,
            "discount": discount,
        }, f)
    df = pandas.DataFrame.from_dict(
        dict(
            rewards=episodic_returns,
            episode=range(training_episodes),
            agent=agent_label,
            seed=seed,
        )
    )
    df['rewards_ma'] = df["rewards"].rolling(window, min_periods=window).mean().shift(-window)

    ax = seaborn.lineplot(data=df, x="episode", y="rewards_ma", hue="agent", ci=None, legend=True)

    seaborn.lineplot(
        data=df, x="episode", y=final_performance,
        ci=None,
        ax=ax,
        linestyle="--",
        legend=True,
        label="Final Policy"
    )

    seaborn.lineplot(
        data=df, x="episode", y=baseline_performance,
        ci=None,
        ax=ax,
        linestyle="-.",
        legend=True,
        label="Behavior Policy"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (moving average {window})")
    ax.set_title(f'{env_id}')
    plt.savefig(f"{output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    train_agent_on_finite_history(**vars(parse_args()))
