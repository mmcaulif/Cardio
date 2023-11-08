#!/bin/bash

# need to: chmod +x run_main.sh, before running

arguments=("1000" "0.01" "0.005" "0.0025")
for arg in "${arguments[@]}"; do
    python main.py exp=standard_run exp.env_steps=500000 alg.target_update="$arg"
done

# estimated time: num_arguments x num_trials x 15mins/100k env steps (15 mins for 100k on laptop)
# 4 x 3 x (5 x 15) = 900mins = 15hours