#!/bin/bash

# need to: chmod +x run_main.sh, before running

arguments=("False" "True")
for arg in "${arguments[@]}"; do
    python main.py exp=standard_run exp.env_steps=3000000 alg.target_update=0.01 alg.env_name=MinAtar/Seaquest-v1 alg.target_selection="$arg"
done
