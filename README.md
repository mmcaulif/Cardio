# GymCardio

## Immediate to do list
* [ ] Implement 'step_for' function in Collector!

* [ ] Create dummy env for testing logging, collection and sampling!

* [ ] Revisit A2C implementation, issue might be in the collector?

## Lower priority
* [ ] Create package!
  * basic implementation done, need to do PyPi and look into further improvements

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