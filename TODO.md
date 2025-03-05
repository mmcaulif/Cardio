# General to do list in advance of an initial 0.1.0 release

# for v0.1.2
1. [ ] Add optional dependencies
1. [ ] Loosen dependency versions
1. [ ] Add agent smoke tests and fakes (similar to toy env)
1. [ ] Nox integration
1. [ ] Verify GymnasiumAtariWrapper works as intended and remove SB3 wrapper (removing SB3 as a requirement).
1. [ ] Allow agents to return metrics from update method to be logged? e.g. Loss
1. [ ] Add recurrent DQN tutorial notebook
1. [ ] Add a system design diagram to readme
1. [ ] Get rid of sprinter and move back to Cardio
1. [ ] Properly document Prioritised Buffer Implementation details
1. [ ] Outline speed, profiling and optimisation roadmap/comparisons
  * [ ] Add function timers and debug mode like in RidL
  * [ ] Compare speed with SB3/SBX

* Follow up with Pablo Samuel Castro

# Post v0.1.2
* [ ] Documentation + white paper
* [ ] Profiling/logging dashboard like PufferLib
* [ ] Ray based vectorised experimenting runner

__Focus on getting some form of Cardio as a finished deliverable__

## Post alpha release
* [ ] Extensive testing!!!

* [ ] Start using Astral UV

* [x] Docker file

* [x] Other replay buffers:
  * [x] trajectory
  * [x] prioritised
  * [x] mixed
  * [x] simple/base

* [ ] Envpool Atari DQN using flax and network resets for Atari 100k benchmarking
  * Will need to make this as quick as possible, use scans for updating over multiple batches
    and XLA environment with scan for evaluation.
