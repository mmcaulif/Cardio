import logging
from typing import Optional

from gymnasium import Env

from cardio_rl import Agent, BaseRunner, Gatherer, VectorGatherer
from cardio_rl.types import Environment

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt=" %I:%M:%S %p",
    level=logging.INFO,
)


class OnPolicyRunner(BaseRunner):
    """The runner is the high level orchestrator that deals with the different
    components and data, it contains a gatherer, your agent and any replay buffer
    you might have. The runner calls the gatherer's step function as part its own
    step function, or as part of its built in warmup (for collecting a large amount
    of initial data with your agent) and burnin (for randomly stepping through an
    environment, not collecting data, such as for initialising normalisation values)
    methods. The runner can either be used via its run method (which iteratively
    calls the runner.step and the agent.update methods) or with each mothod individually
    with its step method if you'd like more finegrained control.

    Attributes:
        env (Environment): The gym environment used to collect transitions and
            train the agent.
        n_envs (int): The number of environments, only > 1 if env is a VectorEnv.
        agent (Optional[Agent], optional): The agent stored within the runner, used
            by the step and run methods. Defaults to None.
        rollout_len (int, optional): Number of environment steps to perform as part
            of the step method. Defaults to 1.
        warmup_len (int, optional): Number of steps to perform with the agent before
            regular rollouts begin. Fixed to 0 for OnPolicyBuffer.
        n_step (int, optional): Number of environment steps to store within a single
            transition. Fixed to 1 for OnPolicyBuffer.
        eval_env (Optional[gym.Env], optional): An optional separate environment to
            be used for evaluation, must not be a VectorEnv. Defaults to None.
        gatherer (Optional[Gatherer], optional): An optional gatherer to be used by
            the runner. Defaults to None.
        _initial_time (float): The time in seconds when the runner was initialised.
    """

    # TODO: use VectorGatherer as the default and explain it, and the differences

    def __init__(
        self,
        env: Environment,
        agent: Optional[Agent] = None,
        rollout_len: int = 1,
        eval_env: Optional[Env] = None,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        """Initialises the runner ...TODO...

        Args:
            env (Environment): The gym environment used to collect transitions and
                train the agent.
            agent (Optional[Agent], optional): The agent stored within the runner, used
                by the step and run methods. Defaults to None.
            rollout_len (int, optional): Number of environment steps to perform as part
                of the step method. Defaults to 1.
            eval_env (Optional[gym.Env], optional): An optional separate environment to
                be used for evaluation, must not be a VectorEnv. Defaults to None.
            gatherer (Optional[Gatherer], optional): An optional gatherer to be used by
                the runner. Defaults to None.
        """
        _gatherer = gatherer or VectorGatherer()
        super().__init__(
            env=env,
            agent=agent,
            rollout_len=rollout_len,
            warmup_len=0,
            n_step=1,
            eval_env=eval_env,
            gatherer=_gatherer,
        )
