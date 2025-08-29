import time

import dm_env
import numpy as np
import reverb

from minasa import specs
from minasa.acme.adders.reverb.sequence import SequenceAdder
from minasa.environments.dm_env_wrappers.single_precision import (
    SinglePrecisionWrapper,
)
from minasa.environments.dm_env_wrappers import create_gym_environment


def run_episodes(env: dm_env.Environment, adder: SequenceAdder, num_episodes: int = 10) -> None:
    action_spec = env.action_spec()
    r = 0.0
    t = 0
    timestep = env.reset()
    adder.add_first(timestep)
    while True:
        a = np.int32(
            np.random.randint(low=0, high=action_spec.num_values)
        )
        timestep = env.step(a)
        adder.add(a, timestep)
        r += timestep.reward
        if timestep.last():
            print(r)
            r = 0.0
            t += 1
            if t >= num_episodes:
                return
            timestep = env.reset()
            adder.add_first(timestep)


env = SinglePrecisionWrapper(create_gym_environment("CartPole-v1"))
environment_spec = specs.make_dmenv_environment_spec(env)

sequence_length = 5

signature = SequenceAdder.signature(
    environment_spec, sequence_length=sequence_length
)

queue = reverb.Table.queue(
    name="table",
    max_size=1024,
    signature=signature,
)

replay_server = reverb.Server([queue], port=None)
replay_client = replay_server.localhost_client()

client = reverb.Client(replay_client.server_address)

adder = SequenceAdder(
    client=client,
    priority_fns={"table": None},
    sequence_length=sequence_length,
    period=sequence_length - 1,
)

print(f"<<< running episodes, pushed to {replay_client.server_address} >>>")
time.sleep(10)

while True:
    time.sleep(1)
    print(f"<<< running episodes, pushed to {replay_client.server_address} >>>")
    run_episodes(env, adder, num_episodes=10)
    # print(queue.info)
    # print(queue.internal_table)
