
# :running: Cardio: Runners for Deep Reinforcement Learning in Gym Environments :running:

<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pythonver](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://www.python.org/doc/versions/)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Style](https://img.shields.io/badge/%20style-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
[![License](https://img.shields.io/badge/python-3.10-blue)](https://github.com/mmcaulif/Cardio/blob/main/LICENSE.txt)

</div>

[Motivation](#motivation) | [Installation](#installation) | [Usage](#usage) | [Under the hood](#under-the-hood) | [Development](#development) | [Contributing](#contributing)

So many reinforcement learning libraries, what makes Cardio different?

* _Easy and readable_: Focus on the agent and leave the boilerplate code to Cardio
* _Extensible_: Easy progression from simple algorithms all the way up to Rainbow and beyond
* _Research friendly_: Cardio was designed to be a whiteboard for your RL research

Cardio aims to make new algorithm implementations easy to do, readable and framework agnostic by providing a collection of modular environment interaction loops for the research and implementation of deep reinforcement learning (RL) algorithms in Gymnasium environments. Out of the box these loops are capable of more complex experience collection approaches such as n-step transitions, trajectories, and storing of auxiliary values to a replay buffer. Accompanying these core components are helpful utilities (such as replay buffers and data transformations), and single-file reference implementations for state-of-the-art algorithms.

## Motivation
In the spectrum of RL libraries, Cardio lies in-between large complete packages such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (lacks modularity/extensibility) that deliver complete implementations of algorithms, and more research-friendly repositories like [CleanRL](https://github.com/vwxyzjn/cleanrl) (repeating boilerplate code), in a similar design paradigm to Google’s [Dopamine](https://github.com/google/dopamine) and [Acme](https://github.com/google-deepmind/acme).

To achieve the desired structure and API, Cardio makes some concessions with the first of which being speed. There's no  competing against end-to-end jitted implementations, but going down this direction greatly hinders the modularity and application of implementations to arbitrary environments. If you are interested in lightning quick training of agents on established baselines then please look towards the likes of [Stoix](https://github.com/EdanToledo/Stoix).

Secondly, taking a modular approach leaves us less immediately extensible than the likes of [CleanRL](https://github.com/vwxyzjn/cleanrl), despite the features in place to make the environment loops transparent, there is inevitably going to be edge cases where Cardio is not the best choice.

## Installation
> **NOTE**: Jax is a major requirement for runner internally, the installation process will be updated soon to make a better distinction between setting up Cardio using Jax for GPU's, CPU's or TPU's. For now please manually install whichever Jax version suits your environment best. By default we just show for cpu but swapping "cpu" out for "gpu" should work all the same.

Prerequisites (to be expanded):
* Python == 3.10


Via pip with Jax cpu:
```bash
pip install cario-rl[cpu]
```

To install is from source via:
```bash
git clone https://github.com/mmcaulif/Cardio.git
cd cardio
pip install ".[cpu]"
```

Alternatively you can install all requirements e.g. for testing, experimenting and development:
```bash
pip install ".[dev,exp,cpu]"
```

Or use the provided makefile (which also sets up the precommit hooks):
```bash
make install_cpu
```

## Usage
Below is a simple exampls (using the CartPole environment) leveraging Cardio's off-policy runner to help write a simple implementation of the core deep RL, Deep Q-Networks. It will be assumed that you have an beginners understanding of deep RL and this section just serves to demonstrate how Cardio might fit into different algorithm implementations.

### DQN
In this algorithm our agent performs a fixed number of environment steps (aka a rollout) and saves the transitions experienced in a replay buffer for performing update steps. Once the rollout is done, we sample from the replay buffer and pass the sampled transitions to the agents update method. To implement our agent we will use the provided Cardio Agent class and override the init, update and step methods:

```python
class DQN(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"lr": 1e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
        use_rmsprop: bool = False,
    ):
        self.env = env
        self.critic = critic
        self.targ_critic = copy.deepcopy(critic)
        self.gamma = gamma
        self.targ_freq = targ_freq
        self.update_count = 0

        if not use_rmsprop:
            self.optimizer = th.optim.Adam(self.critic.parameters(), **optim_kwargs)
        else:
            # TODO: fix mypy crying about return type
            self.optimizer = th.optim.RMSprop(self.critic.parameters(), **optim_kwargs)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

    def update(self, batches):
        data = jax.tree.map(th.from_numpy, batches)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        q = self.critic(s).gather(-1, a)
        q_p = self.targ_critic(s_p).max(dim=-1, keepdim=True).values
        y = r + self.gamma * q_p * ~d

        loss = F.mse_loss(q, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.targ_freq == 0:
            self.targ_critic.load_state_dict(self.critic.state_dict())

        return {}

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state)
            action = self.critic(th_state).argmax().numpy(force=True)
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}
```

Next we instantiate our runner. When we instantiate a runner we will pass it our environment, our agent, rollout length, and the batch size, but there also other arguments you may want to tweak.

```python
env = gym.make("CartPole-v1")
runner = crl.OffPolicyRunner(
    env=env,
    agent=DQN(env, Q_critic(4, 2)),
    rollout_len=4,
    batch_size=32,
)
```

And finally, to run 50,000 rollouts (in this case, 50,000 x 4 environment steps) and perform an agent update after each one, we just use the run method:

```python
runner.run(rollouts=50_000, eval_freq=1_250)
```


### Sprinter
 components are likely to be better suited to Cardio's sibling repo, [Sprinter](https://github.com/mmcaulif/Sprinter) acts as an extension of Cardio, applying the library to create a zoo of different algortihm implementations, and providing simple boilerplate code examples for research focussed tasks such as hyperparameter optimisation or benchmarking, and intended to be cloned/forked and built on as opposed to pip installed.

Sprinter is further behind in its development and currently just acts as a collection of modern algorithm implementations using Jax/Rlax/Flax/Optax.

If you are looking for some more practical examples of Cardio in use (or just an assortment of Jax algorithm implementations),  components are likely to be better suited to Cardio's sibling repo, [Sprinter](https://github.com/mmcaulif/Sprinter) should be all you need.


## Under the hood
Below we'll go over the inner workings of Cardio. The intention was to make Cardio quite minimal and easy to parse, akin to [Dopamine](https://github.com/google/dopamine), but I hope it is interesting to practitioners and I'm eager to hear any feedback/opinions on the design paradigm. This section also serves to highlight a couple of the nuances of Cardio's components.

> **Diagram pending creation**

### Transition
<!-- Italicise the mdp variables !!! -->
Borrowing an idea from [TorchRL](https://github.com/pytorch/rl), the core building block that Cardio centers around is a dictionary that represents an MDP transition. By default the transition dict has the following keys: _s_, _a_, _r_, _s\_p_, _d_ corresponding to _state_, _action_, _reward_, _state'_ (state prime or next state) and _done_. Two important concepts to be aware of are:

1. A Cardio Transition dictionary does not neccessarily correspond to a a single environment step. For example, in the case of n-step transitions _s_ will correspond to _s\_t_ but _s\_p_ will correspnd to _s\_(t+n)_ with the reward key having _n_ number of entries. Furthermore, the replay buffer stores data as a transition dictionary with keys pointing to multiple states, actions rewards etc.
2. The done value used in Cardio is the result of the OR between the terminal and truncated values used in gymnasium. Empiraclly, decoupling termination and truncation has been shown to have a negligible affect. However, this is a trivial feature to change and its possible that leaving up to the user is best.

By using dictionaries, new entries are easy to add and thus the storing of user-defined variables (such as intrinsic reward or policy probabilities) is built in to the framework, whereas this would be nontrivial to implement in more abstract libraries like [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

### Agent
Much like [Acme](https://github.com/google-deepmind/acme) the Cardio agent class is very minimal, simply defining some base methods that are used by the environment interaction loops. The most important thing to know is when they are called, what data is provided, and which component is calling it. The most important of which are the step (given a state, return an action and any extras), view (given a step transition, return any extras) and update methods (given a batch of transitions).

### Gatherer
The gatherer is the primary component in Cardio and serves the purpose of stepping through the environment directly with a provided agent, or a random policy. The gatherer has two buffers that are used to package the transitions for the Runner in the desired manner. The step buffer collects transitions optained from singular environment steps and has a capacity equal to _n_. When the step buffer is full, it transforms its elements into one n-step transition and adds that transition to the transition buffer. Some rough pseudocode is provided below.

<p align="center">
    <a href="docs/images/cardio_gather_pseudocode.png">
        <img src="docs/images/cardio_gather_pseudocode.png" alt="Gatherer pseudocode" width="80%"/>
    </a>
</p>

The step buffer is emptied after terminal states to prevent transitions overlapping across episodes. When _n_ > 1, the step buffer needs to be "flushed", i.e. create transitions from steps that would otherwise be thrown away. Please refer to the example below provided by my esteemed colleage, ChatGPT:

> If you are collecting 3-step transitions, here's how you handle the transitions where s_3 is a terminal state:
> 1. __Transition from s\_0__: (s_0, a_0, [r_0, r_1, r_2], s_3)
> 1. __Transition from s\_1__: (s_1, a_1, [r_1, r_2], s_3)
> 1. __Transition from s\_2__: (s_2, a_2, r_2, s_3)

The transition buffer is even simpler, just containing the processed transitions from the step buffer. The transition buffer starts empty when the gatherer's step method is called and also maintains its data across terminal steps. Both of these characteristics are opposite to the step buffer which persists across gatherer.step calls but not across terminal steps.

Due to the nature of n-step transitions, sometimes the gatherer's transition buffer will have less transitions than environment steps taken (as the step buffer gets filled) and other times it will have more (when the step buffer gets flushed) but at any given time there will be a rough one-to-one mapping between environment steps taken and transitions collected. Lastly, rollout lengths can be less than _n_.

### Runner
The runner is the high level orchestrator that deals with the different components and data, it contains a gatherer, your agent and any replay buffer you might have. The runner step function calls the gatherer's step function as part its own step function, or as part of its built in warmup (for collecting a large amount of initial data with your agent) and burnin (for randomly stepping through an environment, not collecting data, such as for initialising normalisation values) methods. The runner can either be used via its run method (which iteratively calls the runner.step and the agent.update methods) or just with its step method if you'd like more finegrained control.


## Development
The main development goal for Cardio will be to make it as fast, easy to use, and extensible as possible. The aim is not to include many RL features or to cater to every domain. Far down the line I could imagine trying to incorporate async runners but that can get messy quickly. However, if you notice any bugs, or have any suggestions or feature requests, user input is greatly appreciated!

Some tentative tasks right now are:
* [ ] Integrated loggers (WandB, Neptune, Tensorboard etc.)
* [ ] Verify GymnasiumAtariWrapper works as intended and remove SB3 wrapper (removing SB3 as a requirement too).
* [ ] Implement seeding for reproducability.
* [ ] Widespread and rigorous testing!

A wider goal is to perform profiling and squash any immediate performance bottlenecks. Wrapping an environment in a Cardio runner should introduce as little overhead as possible.

Any RL components are likely to be better suited to Cardio's sibling repo, [Sprinter](https://github.com/mmcaulif/Sprinter).

## Contributing
<p align="center">
    <a href="docs/images/cat_pr_image.jpg">
        <img src="docs/images/cat_pr_image.jpg" alt="Cat pull request image" width="40%"/>
    </a>
</p>
Jokes aside, given the roadmap decsribed above for Cardio, PR's related to bugs and performance are the main interest. If you would like a new feature, please create an issue first and we can discuss.

## License
This repository is licensed under the [Apache 2.0 License](https://github.com/mmcaulif/GymCardio/blob/main/LICENSE.txt)
