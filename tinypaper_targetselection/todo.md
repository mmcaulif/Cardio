
Small code for tiny paper submission for target network action selection to combat policy churn
Example DQN: https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
MinAtar paper: https://arxiv.org/pdf/1903.03176.pdf
[x] Install MinAtar 
[x] Setup conv net architecture for Q-func
[x] List hyperparams
[x] Create config (base config + override configs per algorithm + environment (1 for MinAtar, 1 for LunarLander))!
[x] Decouple main and trial function
[ ] Run experiments (DQN first)!
    [ ] Using 1 env, Breakout, conduct initial experiments (500k-1mil timesteps, ~3 trials) to compare soft and hard updates
    [ ] Then compare initial performance boost when using target network action selection based off chosen values
        

Next:
[ ] Do the same for SAC and TD3