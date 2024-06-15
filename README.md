# GymCardio
Cardio offers to replace much of the boiler plate code in deep reinforcement learnign algorithms; namely the data gathering and environment interaction, but also providing pre-written policies and replay buffers, all while aiming to do so in a modular fashion and allow users to implement their own algorithms or improvements to existing algorithms.

The purpose of this library was initially to speed up my own self-implemented versions of algorithms and reducing the code overhead while also reducing the actual number of lines written per script for each algorithm. You'll often find that the actual algorithm details often corresponds to very little code and handling of the environment and buffer causes even simple implementations to feel somewhat bloated. Thus cardio hopes to allow for simple, easy to read, one file implementations. In an effort to showcase the library I have included a number of 'stubs' that are just that.

This is still heavily a work in progress and even many of the (poorly organised) stubs do not work with the current versions of the runner. Going forward I will be chipping away at streamlining, organising and documenting this repository when I get the chance!

# Cardio basics
This section will be overhauled at a later date, but now the gist of Cardio is the Runner class that gives you a simple interface that wraps the environment and then with one method, will step through the environment, collect transitions, and process them in your defined way. The runner supports n-step transitions, custom policies and custom processing. Currently the runner class is biased towards off-policy algorithms in mind (altough fully supporting on-policy approaches too) but going forward it will be better balanced between the two.

# Big Refactor and prerequisites for release

__TLDR summary of tasks:__
1. create a base transition dict/dataclass/tree
2. write a replay buffee that uses these
3. extend that to the internals of the gatherer

Think of the replay buffer and internal buffers as just larger/longer versions of the transition tree datatype, akin to TorchRL

* [ ] Refactor Gatherer and Runners, keep minimalist and introduce an agent class
  * [x] Get rid of the use of "__call__" methods for runner etc. use .step() and .run() instead
  * [x] Add easier appending to replay buffer, no for loops (specified below, implement a pytree replay buffer)
  * [x] Cleaner passing of rollout step and warmup length handling
  * [ ] Review Gatherer inner workings and runner inner workings
    * [ ] You'll absolutely need to document these well and understand them well (incl. edge cases)
  * [x] Add evaluation methodology
  * [x] Add reset, update_agent and load methods for runner/agent (e.g. for use in Reptile impl)
  * [x] Add n-step collection

* [ ] QOL Runner and Gatherer changes
  * [x] Add num for buffer's store method to part of runner/gatherer, instead of manually calculated
  * [ ] Write up documents and doc strings for Runner and Gatherer to make it easier to understand!!!
    * Use google style docstrings
  * [ ] Verify that Runner can be used without supplying an agent in a manner as expected
    * an agent will need to be initially supplied for warmup but allow for it afterwards
  * [ ] Implement sequence/trajectory collection to runner/gatherer, have it be compatible with n-step
    * This is can be done via the replay buffer most likely!

* [ ] Improve logging
  * [x] Current time spent
  * [x] Env steps per second
  * [ ] Move logging from gatherer (will still need to maintain some logging in gatherer like episodes etc.)
  * [ ] Make logger/metrics system extensible (use dictionaries to pass around)
  * [ ] Rich logging, make it pretty and formatted well!!!
  * [ ] Explore if logging could be done outside the gatherer (as its very nested)
  * [ ] Figure out a way to make logging extensible and customisable
  * [ ] Gatherer should return the number of steps taken and episodes completed

* [ ] Improve extensibility
  * [x] Agents should be able to use and save extras (such as log probs)
    * add indeces sampled to batch data outputted
  * [x] Implement a pytree based replay buffer with saving of multiple transitions in parallel
    * this ties into the above with using pytree's internally within the gatherer
  * [ ] Implement dopamine Rainbow as an example
    * [ ] Add buffer overriding via extra info from agent update method

* [ ] Make library presentable
  * [ ] Jax agent stubs
    * [ ] Will need to get access to a linux machine with a GPU to properly evaluate performance/speed
    * [ ] Consider different popular algorithms and how they could be implemented easily
    * [ ] MinAtar baselines
  * [ ] Simple examples
    * [ ] DDPG
    * [ ] SAC
    * [ ] TD3
    * [ ] Simplified MPO (?)
    * [ ] Munchausen-DQN
    * [ ] Dopamine Rainbow
    * [ ] V-trace with experience replay
  * [ ] Intermediate examples
    * [ ] Simple NGU
    * [ ] Simple SPR
    * [ ] Simple DrQ(eps)
    * [ ] Data efficient Rainbow
    * [ ] HL-Gauss
    * [ ] Soft MOE
  * [x] Linting and typing
  * [ ] Readme and docs, look at stoicx for inspo
  * [ ] Extensive and widespread testing!!!
  * [x] Precommit hooks
  * [x] Make file
  * [ ] Docker file (?)

__Focus on getting some form of Cardio as a finished deliverable__

## Post-release
* [ ] Github integrations for ruff, type checking
* [ ] More test coverage and github integration


## To do list
* [ ] Make default research template via make, conda, hydra, optuna etc.
* [ ] Minor refactor to gatherer and runner, add default arg values, careful consideration needed
  * [x] change collector name to gatherer, idk why its different
  * still not confident on all changes and structure
* [ ] Create package!
  * [x] basic implementation done
  * [ ] need to do PyPi and look into further improvements
  * [ ] documentation, contribution guide, linting etc.
* [x] Remove warmup method in gatherer and make it a special call of the rollout method
* [ ] Add trajectory collection
  * [ ] Recurrent DQN
* [x] Add all action spaces to circular buffer
* [ ] Benchmark and debug after restructuring


## Vectorised env work
* [ ] Align VectorCollector with Collector
  * Revisit A2C implementation
  * Make Logger compatible with VectorCollector
  * Learning doesnt seem to line up with stable baselines3 need to debug all aspects (collector, value estimation and logger)

* [ ] Makes sure vector collector work as intended for off-policy methods and n-step collector work as intended for on-policy methods etc.

## Longer horizon priority
* [ ] Benchmark each implementation wrt. SB3 (change logging to timestep based first though)

* [ ] Make library faster and performant!!!

* [ ] Offline gatherer
  * on pause until mujoco sorted

* [ ] A modern sb3-style collection of baselines as a submodule

## Completed
* [x] Implement multibatch sampling for off-policy runner

* [x] Add episode length to logger and use the same names as SB3 for easy integration!

* [x] Change logging from episodic to timestep based
  * include window and log_interval arguments to gatherer

* [x] Implement 'step_for' function in Collector!

* [x] Create dummy env for testing logging, collection and sampling!

* [x] Implement 'reduce' argument for n-step learning that returns unsqueezed vectors (for DRQN)
