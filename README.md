# GymCardio
Cardio offers to replace much of the boiler plate code in deep reinforcement learnign algorithms; namely the data gathering and environment interaction, but also providing pre-written policies and replay buffers, all while aiming to do so in a modular fashion and allow users to implement their own algorithms or improvements to existing algorithms.

The purpose of this library was initially to speed up my own self-implemented versions of algorithms and reducing the code overhead while also reducing the actual number of lines written per script for each algorithm. You'll often find that the actual algorithm details often corresponds to very little code and handling of the environment and buffer causes even simple implementations to feel somewhat bloated. Thus cardio hopes to allow for simple, easy to read, one file implementations. In an effort to showcase the library I have included a number of 'stubs' that are just that.

This is still heavily a work in progress and even many of the (poorly organised) stubs do not work with the current versions of the runner. Going forward I will be chipping away at streamlining, organising and documenting this repository when I get the chance!

# Cardio basics
This section will be overhauled at a later date, but now the gist of Cardio is the Runner class that gives you a simple interface that wraps the environment and then with one method, will step through the environment, collect transitions, and process them in your defined way. The runner supports n-step transitions, custom policies and custom processing. Currently the runner class is biased with off-policy algorithms in mind (altough fully supporting on-policy approaches too) but going forward it will be better balanced between the two.

## Immediate to do list
* [ ] Add timing information to logger

* [ ] Minor refactor to gatherer and runner, add default arg values, careful consideration needed
  * mostly done, just some final decisions to make
  * change collector name to gatherer, idk why its different

* [ ] Create package!
  * basic implementation done, need to do PyPi and look into further improvements

* [x] Implement replay buffer class and move IET work into Cardio ecosystem

* [ ] Remove warmup method in gatherer and make it a special call of the rollout method

## Vectorised env work
* [ ] Align VectorCollector with Collector
  * Revisit A2C implementation
  * Make Logger compatible with VectorCollector
  * Learning doesnt seem to line up with stable baselines3 need to debug all aspects (collector, value estimation and logger)

* [ ] Makes sure vector collector work as intended for off-policy methods and n-step collector work as intended for on-policy methods etc.

## Lower priority
* [ ] Offline gatherer
  * on pause until mujoco sorted

* [ ] Move agent stubs into own folder and refactor each one

* [ ] Sort policies better, i.e. discrete, continuous

* [ ] Benchmark each implementation wrt. SB3 (change logging to timestep based first though)
 
## Completed
* [x] Implement multibatch sampling for off-policy runner

* [x] Add episode length to logger and use the same names as SB3 for easy integration!

* [x] Parallel gatherer

* [x] Change logging from episodic to timestep based
  * include window and log_interval arguments to gatherer

* [x] Implement 'step_for' function in Collector!

* [x] Create dummy env for testing logging, collection and sampling!

* [x] Implement 'reduce' argument for n-step learning that returns unsqueezed vectors (for DRQN)