# To do list
* [ ] Benchmark TD3 implementation with SB3!
* [ ] set up proper experimenting pipeline, benchmark TD3 implementation etc.
* [x] add central buffer that all transitions are added to and create a sampling ratio between central buffer and event tables
* [ ] implement encoder architecture and gradient updates
    * [x] Siamese action regression network (only state)
    * [ ] VAE (can be everything)
    * [ ] TD7-like (state and action)

# Notes
* Platueaing could be due to replay buffer being too large, SB3 uses a size of 200,000 for BipedalWalker, test this hypothesis with SB3