#!/bin/bash

# need to: chmod +x run_main.sh, before running

arguments=("0.0025" "0.01") # ("1000" "0.005")
for arg in "${arguments[@]}"; do
    python main.py exp=standard_run exp.n_trials=2 exp.name=tau_"$arg" exp.env_steps=1000000 alg.target_update="$arg" alg.env_name=MinAtar/Asterix-v1
done
