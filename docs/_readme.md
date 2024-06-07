
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![Pythonver](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://www.python.org/doc/versions/)

[![License](https://img.shields.io/badge/python-3.10-blue)](https://github.com/mmcaulif/GymCardio/blob/main/LICENSE.txt)

</div>

<h2 align="center">
    <p>Cardio: Runners for Deep Reinforcement Learning in Gym Environments</p>
</h2>

<div align="center">

**_Cleaner_ RL**

</div>

<!-- Overview -->
- ⚡️
- 🐍
- 🛠️
- 🤝
- ⚖️
- 📦
- 🔧
- 📏
- ⌨️
- 🌎

_general aims and bit about cardio_
<!-- End overview -->

## Table of Contents
1. [Cardio overview](#cardio-overview)
1. [Installation](#installation)
1. [Motivation](#motivation)
1. [Under the hood](#under-the-hood)
1. [Simple Examples](#simple-examples)
1. [Intermediate Examples](#intermediate-examples)
1. [Development](#development)
1. [Contributing](#contributing)

## Cardio overview
Cardio aims to make new algorithm implementations easy, readable and framework agnostic by providing a collection of modular environment interaction loops for the research and implementation of deep reinforcement (RL) algorithms in Gymnasium environments. By default these loops are capable of more complex experience collection approaches such as n-step transitions, trajectories, and storing of auxiliary values to a replay buffer. Accompanying these core components are helpful utilities (such as replay buffers and data transformations), and single-file reference implementations for state-of-the-art algorithms.


## Installation
> **NOTE**: Jax is a major requirement both internally and also for the agent implementations, the installation process will be updated soon to make a better distinction between setting up Cardio using Jax for GPU's, CPU's or TPU's. For now the default is CPU but feel free to use whichever.

Prerequisites:
* Python == 3.10

For now, the way to install is from source via:
```bash
git clone https://github.com/mmcaulif/GymCardio.git
cd gymcardio
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
In the spectrum of RL libraries, Cardio lies in-between large complete packages such as stable-baselines3 (lacks modularity/extensibility) that deliver complete implementations of algorithms, and more research-friendly repositories like CleanRL (non-algorithm boilerplate code), in a similar design paradigm to Google’s Dopamine and Acme.

To achieve the desired structure and API, Cardio makes some concessions with the first of which being speed. There's no  competing against end-to-end jitted implementations, but going down this direction greatly hinders the modularity and application of implementations to arbitrary environments. If you are interested in lightning quick training of agents on established baselines then please look towards the likes of Stoix.

Secondly, taking a modular approach leaves us less immediately extensible than the likes of CleanRL, despite the features in place to make the environment loops transparent, there is inevitably going to be edge cases where Cardio is not the best choice.


## Under the hood

## Simple Examples

### Q-Learning

### Reinforce

### DQN

## Intermediate Examples

## Development

## Contributing
<!-- You'll need to change the relative path once making this the actual readme -->
<p align="center">
    <a href="images/cat_pr_image.jpg">
        <img src="images/cat_pr_image.jpg" alt="Cat pull request image" width="30%"/>
    </a>
</p>

## License
This repository is licensed under the [Apache 2.0 License](https://github.com/mmcaulif/GymCardio/blob/main/LICENSE.txt)
