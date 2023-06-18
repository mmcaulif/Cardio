# GymCardio

## Immediate to do list

* [ ] Create dummy env for testing logging, collection and sampling!

* [x] Parallel gatherer
  * think it works, need to debug and implement A2C!

* [ ] Create package!
  * basic implementation done, need to do PyPi and look into further improvements

* [ ] Offline gatherer
  * on pause until mujoco sorted

* [ ] Refactor gatherer and runner, add default arg values, careful consideration needed
  * mostly done, just some final decisions to make

* [ ] Move agent stubs outside of src and using package instead of src

* [ ] Change logging from episodic to timestep based
  * include window and print freq arguments to gatherer

* [ ] Sort policies better, i.e. discrete, continuous

* [ ] Benchmark each implementation wrt. SB3 (change logging to timestep based first though)
 