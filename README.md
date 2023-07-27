# GymCardio

## Immediate to do list
* [ ] Create package!
  * basic implementation done, need to do PyPi and look into further improvements

* [ ] Implement multibatch sampling for off-policy runner

* [ ] Add episode length to logger and use the same names as SB3 for easy integration!

* [ ] Align VectorCollector with Collector
  * Revisit A2C implementation
  * Make Logger compatible with VectorCollector
  * Learning doesnt seem to line up with stable baselines3 need to debug all aspects (collector, value estimation and logger)

* [ ] Makes sure vector collector work as intended for off-policy methods and n-step collector work as intended for on-policy methods etc.

## Lower priority

* [ ] Offline gatherer
  * on pause until mujoco sorted

* [ ] Minor refactor to gatherer and runner, add default arg values, careful consideration needed
  * mostly done, just some final decisions to make
  * change collector name to gatherer, idk why its different

* [ ] Move agent stubs outside of src and using package instead of src

* [ ] Sort policies better, i.e. discrete, continuous

* [ ] Benchmark each implementation wrt. SB3 (change logging to timestep based first though)
 
## Completed
* [x] Parallel gatherer

* [x] Change logging from episodic to timestep based
  * include window and log_interval arguments to gatherer

* [x] Implement 'step_for' function in Collector!

* [x] Create dummy env for testing logging, collection and sampling!

* [x] Implement 'reduce' argument for n-step learning that returns unsqueezed vectors (for DRQN)