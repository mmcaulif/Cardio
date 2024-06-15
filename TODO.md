# General to do list in advance of an initial 0.0.1 release

## Primary goals rn
1. Documentation (docstrings, readme, basic examples)
    * describe the edgecases accounted for in the code (i.e. n-step transitions and terminal steps)
    * describe the vocabulary used in the code base
1. Testing (rigorous testing of gatherer and runner)
1. clean up repo but be sure to keep dev and WIP features in a branch

Once done with the 0.0.1 version, send to different people for feedback

## Specific tasks
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
    * add indices sampled to batch data outputted (?)
  * [x] Implement a pytree based replay buffer with saving of multiple transitions in parallel
    * this ties into the above with using pytree's internally within the gatherer
  * [ ] Implement dopamine Rainbow as an example
    * [ ] Add buffer overriding via extra info from agent update method

* [ ] Make library presentable
  * [ ] Simple examples
  * [x] Linting and typing
  * [ ] Readme and docs, look at stoix for inspo
    * [ ] Pseudocode for gatherer and runner internals
  * [ ] Extensive and widespread testing!!!
  * [x] Precommit hooks
  * [x] Make file
  * [ ] Docker file (?)
  * [ ] Jax agent stubs
    * [ ] Will need to get access to a linux machine with a GPU to properly evaluate performance/speed
    * [ ] Consider different popular algorithms and how they could be implemented easily
    * [ ] MinAtar baselines

__Focus on getting some form of Cardio as a finished deliverable__

## Post-release
* [ ] Github integrations for ruff, type checking
* [ ] More test coverage and github integration
