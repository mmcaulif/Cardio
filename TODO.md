# General to do list in advance of an initial 0.0.1 release

## Primary goals
1. [ ] Testing (rigorous testing of gatherer and runner)
  * [ ] Parameterised testing
1. [x] clean up repo but be sure to keep WIP features in a branch
1. [x] decouple the actions into seperate files per action (?)
1. [x] add gatherer pseudocode to readme
1. [ ] finish readme and docstrings
  * [ ] contributing
  * [x] examples, include code snippets
  * [x] development
  * [x] write a better description
  * [ ] look at other repo's for inspo
  * [x] add some emoji's for fun
1. [ ] add custom templates for issues and PR's
  * look at other repo's for inspo and give credit
1. [ ] add a logo to top of readme
  * A robot version of the running man emoji maybe? Or something similar
1. [ ] Final review and draft of Readme

Once done with the 0.1.0 version, send to different people for feedback

## Specific tasks
* [ ] QOL Runner and Gatherer changes
  * [x] Add num for buffer's store method to part of runner/gatherer, instead of manually calculated
  * [x] Verify that Runner can be used without supplying an agent in a manner as expected
    * an agent will need to be initially supplied for warmup but allow for it afterwards
  * [ ] Make it so if an agent isnt passed initially, the warmup will use a random policy (necessary?)

* [ ] Improve extensibility
  * [x] Agents should be able to use and save extras (such as log probs)
    * add indices sampled to batch data outputted (?)
  * [x] Implement a pytree based replay buffer with saving of multiple transitions in parallel
    * this ties into the above with using pytree's internally within the gatherer
  * [ ] Implement dopamine Rainbow as an intermediate example

* [ ] Make library presentable
  * [ ] Doc strings for Runner, Gatherer and other components to make it easier to understand!!!
    * Use google style docstrings
  * [ ] Readme and docs, look at stoix for inspo
    * [x] Pseudocode for gatherer internals
  * [ ] Extensive testing!!!
  * [ ] Jax agent stubs
    * [ ] Will need to get access to a linux machine with a GPU to properly evaluate performance/speed
    * [ ] Consider different popular algorithms and how they could be implemented easily

* [ ] While trying to implement A2C and PPO, you broke the runner/gatherer, it is mostly fixed
      but double check everything! Further proof that testing is needed...

* [ ] MinAtar and Atari examples
  * [x] MinAtar: Seems to work seamlessly so far, need to write a network for it and train DQN
    * MinAtar DQN appears to match performance from paper in initial benchmarks of Freeway!
  * [ ] Atari:
  * [ ] Envpool:

* Maybe make evaluation done within an agent, the runner can pass it metrics like steps taken or
  episodes completed. Will need to be experimented with...

__Focus on getting some form of Cardio as a finished deliverable__

## Post alpha release
* [ ] Pip package with github actions for releases

* [ ] Docker file

* [ ] Other replay buffers:
  * [ ] trajectory
  * [x] prioritised
  * [ ] mixed
  * [ ] simple/base

* [ ] Envpool Atari DQN using flax and network resets for Atari 100k benchmarking
  * Will need to make this as quick as possible, use scans for updating over multiple batches
    and XLA environment with scan for evaluation.

* [ ] Outline benchmarking roadmap
  * [ ] Make seperate repo
* [ ] Outline speed, profiling and optimisation roadmap

* [ ] Make seperate repo for research/experimenting template

* [ ] Add a system design diagram to readme

* [ ] Improve logging
  * [x] Current time spent
  * [x] Env steps per second
  * [ ] Move logging from gatherer (will still need to maintain some logging in gatherer like episodes etc.)
  * [ ] Make logger/metrics system extensible (use dictionaries to pass around)
  * [ ] Rich logging, make it pretty and formatted well!!!
  * [ ] Explore if logging could be done outside the gatherer (as its very nested)
  * [ ] Figure out a way to make logging extensible and customisable
  * [ ] Gatherer should return the number of steps taken and episodes completed

# Done
* [x] Simple examples
* [x] Linting and typing
* [x] Precommit hooks
* [x] Make file
* [x] Refactor Gatherer and Runners, keep minimalist and introduce an agent class
  * [x] Get rid of the use of "__call__" methods for runner etc. use .step() and .run() instead
  * [x] Add easier appending to replay buffer, no for loops (specified below, implement a pytree replay buffer)
  * [x] Cleaner passing of rollout step and warmup length handling
  * [x] Review Gatherer inner workings and runner inner workings
    * You'll absolutely need to document these well and understand them well (incl. edge cases)
  * [x] Add evaluation methodology
  * [x] Add reset, update_agent and load methods for runner/agent (e.g. for use in Reptile impl)
  * [x] Add n-step collection

* [x] Github integrations for ruff, type checking
* [x] More test coverage and github integration
