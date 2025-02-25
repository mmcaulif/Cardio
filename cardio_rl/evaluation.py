"""Toy Environment for debugging."""

import time

from gymnasium import Env  # type: ignore
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from cardio_rl.agent import Agent  # type: ignore


def evaluate_agent(
    env: Env, agent: Agent, episodes: int, return_episode_rewards: bool = False
) -> tuple[list | float, float]:
    """TODO.

    TODO.

    Args:
        env (Env): _description_
        agent (Agent): _description_
        episodes (int): _description_
        return_episode_rewards (bool, optional): _description_. Defaults to False.

    Returns:
        Union[tuple[list, float], tuple[float, float]]: Episode rewards and time taken.
    """
    assert isinstance(env, RecordEpisodeStatistics)

    avg_r = []
    t = time.time()
    for _ in range(episodes):
        s, _ = env.reset()
        while True:
            a = agent.eval_step(s)
            s_p, _, term, trun, info = env.step(a)
            done = term or trun
            s = s_p
            if done:
                avg_r.append(info["episode"]["r"])
                break

    total_t = time.time() - t
    if return_episode_rewards:
        return avg_r, total_t
    else:
        avg_r = sum(avg_r) / episodes
        return avg_r, total_t
