
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pythonver](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://www.python.org/doc/versions/)
[![License](https://img.shields.io/badge/python-3.10-blue)](https://github.com/mmcaulif/GymCardio/blob/main/LICENSE.txt)

</div>

<h2 align="center">
    <p>Cardio: Runners for Deep Reinforcement Learning in Gym Environments</p>
</h2>

<div align="center">

**Tidy RL**

</div>

> **NOTE**: _note about cardio being made for myself first and foremost, and being a WIP_

## Cardio overview
Cardio aims to make new algorithm implementations easy to do, readable and framework agnostic by providing a collection of modular environment interaction loops for the research and implementation of deep reinforcement (RL) algorithms in Gymnasium environments. By default these loops are capable of more complex experience collection approaches such as n-step transitions, trajectories, and storing of auxiliary values to a replay buffer. Accompanying these core components are helpful utilities (such as replay buffers and data transformations), and single-file reference implementations for state-of-the-art algorithms.

## Table of Contents
1. [Installation](#installation)
1. [Motivation](#motivation)
1. [Simple Examples](#simple-examples)
1. [Under the hood](#under-the-hood)
1. [Intermediate Examples](#intermediate-examples)
1. [Development](#development)
1. [Contributing](#contributing)


## Installation
> **NOTE**: Jax is a major requirement both internally and also for the agent implementations, the installation process will be updated soon to make a better distinction between setting up Cardio using Jax for GPU's, CPU's or TPU's. For now the default is CPU but feel free to use whichever.

Prerequisites:
* Python == 3.10

For now, the way to install is from source via:
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


## Motivation
In the spectrum of RL libraries, Cardio lies in-between large complete packages such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (lacks modularity/extensibility) that deliver complete implementations of algorithms, and more research-friendly repositories like [CleanRL](https://github.com/vwxyzjn/cleanrl) (repeating boilerplate code), in a similar design paradigm to Google’s [Dopamine](https://github.com/google/dopamine) and [Acme](https://github.com/google-deepmind/acme).

To achieve the desired structure and API, Cardio makes some concessions with the first of which being speed. There's no  competing against end-to-end jitted implementations, but going down this direction greatly hinders the modularity and application of implementations to arbitrary environments. If you are interested in lightning quick training of agents on established baselines then please look towards the likes of [Stoix](https://github.com/EdanToledo/Stoix).

Secondly, taking a modular approach leaves us less immediately extensible than the likes of [CleanRL](https://github.com/vwxyzjn/cleanrl), despite the features in place to make the environment loops transparent, there is inevitably going to be edge cases where Cardio is not the best choice.


## Simple Examples

### Q-Learning

### Reinforce

### DQN

## Under the hood
Below we'll go over the inner workings of Cardio. The intention was to make Cardio quite minimal and easy to parse, akin to [Dopamine](https://github.com/google/dopamine), but I hope it is interesting to practitioners and I'm eager to hear any feedback/opinions on the design paradigm. This section also serves to highlight a couple of the nuances of Cardio's components.

> **Diagram pending creation**

### Transition
<!-- Italicise the mdp variables !!! -->
Borrowing an idea from [TorchRL](https://github.com/pytorch/rl), the core building block that Cardio centers around is a dictionary that represents an MDP transition. By default the transition dict has the following keys: _s_, _a_, _r_, _s\_p_, _d_ corresponding to _state_, _action_, _reward_, _state'_ (state prime or next state) and done. Two important concepts to be aware of are:

1. A Cardio Transition dictionary does not neccessarily correspond to a a single environment step. For example, in the case of n-step transitions s will correspond to _s\_t_ but _s\_p_ will correspnd to _s\_(t+n)_ with the reward key having _n_ number of entries. Furthermore, the replay buffer stores data as a transition dictionary with keys pointing to multiple states, actions rewards etc.
2. The done value used in Cardio is the result of the OR between the terminal and truncated values used in gymnasium. Empiraclly, decoupling termination and truncation has been shown to have a negligible affect. However, this is a trivial feature to change and its possible that leaving up to the user is best.

By using dictionaries, new entries are easy to add and thus the storing of user-defined variables (such as intrinsic reward or policy probabilities) is built in to the framework, whereas this would be nontrivial to implement in more abstract libraries like [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

### Agent
Much like [Acme](https://github.com/google-deepmind/acme) the Cardio agent class is very minimal, simply defining some base methods that are used by the environment interaction loops. The most important thing to know is when they are called, what data is provided, and which component is calling it. The most important of which are the step (given a state, return an action and any extras), view (given a step transition, return any extras) and update methods (given a batch of transitions).

### Gatherer
The gatherer is the primary component in Cardio and serves the purpose of stepping through the environment directly with a provided agent, or a random policy. The gatherer has two buffers that are used to package the transitions for the Runner in the desired manner. The step buffer collects transitions optained from singular environment steps and has a capacity equal to _n_. When the step buffer is full, it transforms its elements into one n-step transition and adds that transition to the transition buffer. The step buffer is emptied after terminal states to prevent transitions overlapping across episodes. When _n_ > 1, the step buffer needs to be "flushed", i.e. create transitions from steps that would otherwise be thrown away. Please refer to the example below provided by my esteemed colleage, ChatGPT:

> If you are collecting 3-step transitions, here's how you handle the transitions where s_3 is a terminal state:
> 1. __Transition from s\_0__: (s_0, a_0, [r_0, r_1, r_2], s_3)
> 1. __Transition from s\_1__: (s_1, a_1, [r_1, r_2], s_3)
> 1. __Transition from s\_2__: (s_2, a_2, r_2, s_3)

The transition buffer is even simpler, just containing the processed transitions from the step buffer. The transition buffer starts empty when the gatherer's step method is called and also maintains its data across terminal steps. Both of these characteristics are opposite to the step buffer which persists across gatherer.step calls but not across terminal steps.

Due to the nature of n-step transitions, sometimes the gatherer's transition buffer will have less transitions than environment steps taken (as the step buffer gets filled) and other times it will have more (when the step buffer gets flushed) but at any given time there will be a rough one-to-one mapping between environment steps taken and transitions collected. Lastly, rollout lengths can be less than _n_.

### Runner
The runner is the high level orchestrator that deals with the different components and data, it contains a gatherer, your agent and any replay buffer you might have. The runner step function calls the gatherer's step function as part its own step function, or as part of its built in warmup (for collecting a large amount of initial data with your agent) and burnin (for randomly stepping through an environment, not collecting data, such as for initialising normalisation values) methods. The runner can either be used via its run method (which iteratively calls the runner.step and the agent.update methods) or just with its step method if you'd like more finegrained control.

## Intermediate Examples
_coming soon_

## Development

## Contributing
<!-- You'll need to change the relative path once making this the actual readme -->
<p align="center">
    <a href="docs/images/cat_pr_image.jpg">
        <img src="docs/images/cat_pr_image.jpg" alt="Cat pull request image" width="40%"/>
    </a>
</p>

## License
This repository is licensed under the [Apache 2.0 License](https://github.com/mmcaulif/GymCardio/blob/main/LICENSE.txt)
