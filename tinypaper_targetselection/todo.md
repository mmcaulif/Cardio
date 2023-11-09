Small code for tiny paper submission for target network action selection to combat policy churn
Example DQN: https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
MinAtar paper: https://arxiv.org/pdf/1903.03176.pdf
Small batch anomaly tiny paper: https://openreview.net/forum?id=G0heahVv5Y
[x] Install MinAtar 
[x] Setup conv net architecture for Q-func
[x] List hyperparams
[x] Create config (base config + override configs per algorithm + environment (1 for MinAtar, 1 for LunarLander))!
[x] Decouple main and trial function
[ ] Maybe try using delayed soft target updates with DQN?
[ ] Run experiments (DQN first)! (for reference, 1mil ~= 40 minutes on fatclient)
    [ ] Using 1 env, Axterix, conduct initial experiments (500k-1mil timesteps, ~3 trials) to compare soft and hard target net updates
        -tau=0.005 has beaten the default hard target network updates, now will try with 0.01 and 0.0025
    [ ] Then compare initial performance boost when using target network action selection based off chosen values (1 environment -
        Seaquest, 3mil, 3 trials, both with same target updates)
        

Next:
[ ] Do the same for SAC and TD3