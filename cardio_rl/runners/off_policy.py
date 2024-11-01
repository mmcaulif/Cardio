"""Deprecated off policy runner class."""

import warnings

from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv

from cardio_rl import Agent, Gatherer
from cardio_rl.buffers import BaseBuffer, TreeBuffer
from cardio_rl.loggers import BaseLogger
from cardio_rl.runners.runner import Runner
from cardio_rl.types import Environment


class OffPolicyRunner(Runner):
    """Construct a Runner for off-policy learning."""

    def __init__(
        self,
        env: Environment,
        agent: Agent | None = None,
        extra_specs: dict = {},
        buffer: BaseBuffer | None = None,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        eval_env: Env | None = None,
        logger: BaseLogger | None = None,
        gatherer: Gatherer | None = None,
    ) -> None:
        """Construct a Runner for off-policy learning.

        This class is deprecated and only included for backward
        compatability. Please use the Runner.off_policy class method
        included in the Runner class.

        Args:
            env (Environment): The gym environment used to collect
                transitions and train the agent.
            agent (Agent | None, optional): The agent stored within the
                runner, used by the step and run methods. If set to
                None, a random Agent is used. Defaults to None.
            extra_specs (dict | None, optional):  Additional entries to
                add to the replay buffer. Values must correspond to the
                shape. Defaults to None. Defaults to None.
            buffer (BaseBuffer | None, optional): Can pass a buffer
                which stores data and is randomly sampled when calling
                runner.step, if set to None a default TreeBuffer will
                be used with kwargs equal to those defined in the
                buffer_kwargs argument. Defaults to None.
            rollout_len (int, optional): Number of environment steps to
                perform as part of the step method. Defaults to 1.
            batch_size (int, optional): Batch size to sample from the
                buffer. If set to None, the sample method will expect
                sample indices to be provided. Defaults to 100.
            warmup_len (int, optional): Number of steps to perform with
                the agent before regular rollouts begin. Defaults to
                10_000.
            n_batches (int, optional): How many batches of batch_size
                samples to do at sample time, requires a batch_size
                be provided. Defaults to 1.
            eval_env (gym.Env | None, optional): An optional separate
                environment to be used for evaluation, must not be a
                VectorEnv. Defaults to None.
            logger (BaseLogger | None, optional): The logger used
                during evaluations. If set to None, uses the BaseLogger
                which prints to terminal and writes to a file in a
                logs directory. Defaults to None.
            gatherer (Gatherer | None, optional): An optional gatherer
                to be used by the runner. If set to None the default
                Gatherer will be used. Defaults to None.

        Raises:
            TypeError: Trying to pass a VectorEnv environment to this
                runner.
        """
        warnings.warn(
            "OffPolicyRunner is deprecated, please use cardio_rl.Runner.off_policy instead"
        )
        del gatherer
        if isinstance(env, VectorEnv):
            raise TypeError("VectorEnv's not yet compatible with off-policy runner")

        if buffer is not None:
            warnings.warn(
                "Providing a buffer, ignoring the extra_specs and buffer_kwargs arguments"
            )
            buffer = buffer
        else:
            buffer = TreeBuffer(
                env, extra_specs=extra_specs, batch_size=batch_size, n_batches=n_batches
            )

        super().__init__(
            env=env,
            agent=agent,
            rollout_len=rollout_len,
            warmup_len=warmup_len,
            n_step=buffer.n_steps,
            eval_env=eval_env,
            buffer=buffer,
            logger=logger,
        )
