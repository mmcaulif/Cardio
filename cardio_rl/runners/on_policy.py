"""Deprecated on policy runner class."""

import warnings

from gymnasium import Env

from cardio_rl import Agent, Gatherer, VectorGatherer
from cardio_rl.loggers import BaseLogger
from cardio_rl.runners.runner import Runner
from cardio_rl.types import Environment


class OnPolicyRunner(Runner):
    """Construct a Runner for on-policy learning."""

    def __init__(
        self,
        env: Environment,
        agent: Agent | None = None,
        rollout_len: int = 1,
        eval_env: Env | None = None,
        logger: BaseLogger | None = None,
        gatherer: Gatherer | None = None,
    ) -> None:
        """Construct a Runner for on-policy learning.

        This class is deprecated and only included for backward
        compatability. Please use the Runner.on_policy class method
        included in the Runner class.

        Args:
            env (Environment): The gym environment used to collect
                transitions and train the agent.
            agent (Agent | None, optional): The agent stored within the
                runner, used by the step and run methods. If set to
                None, a random Agent is used. Defaults to None.
            rollout_len (int, optional): Number of environment steps to
                perform as part of the step method. Defaults to 1.
            eval_env (Env | None, optional): An optional separate
                environment to be used for evaluation, must not be a
                VectorEnv. Defaults to None.
            logger (BaseLogger | None, optional): The logger used
                during evaluations, is set to None uses the BaseLogger
                which prints to terminal and writes to a file in a
                logs directory. Defaults to None.
            gatherer (Gatherer | None, optional): An optional gatherer
                to be used by the runner. If set to None the
                VectorGatherer will be used. Defaults to None.
        """
        warnings.warn(
            "OnPolicyRunner is deprecated, please use cardio_rl.Runner.on_policy instead"
        )
        _gatherer = gatherer or VectorGatherer()
        super().__init__(
            env=env,
            agent=agent,
            rollout_len=rollout_len,
            warmup_len=0,
            n_step=1,
            eval_env=eval_env,
            logger=logger,
            gatherer=_gatherer,
        )
