#!/bin/bash

# need to: chmod +x run_main.sh, before running
environments=("MinAtar/Breakout-v1" "MinAtar/Seaquest-v1" )
for env in "${environments[@]}"; do
    arguments=("False" "True")
    for arg in "${arguments[@]}"; do
        python main.py exp=standard_run alg.target_update=0.005 alg.env_name="$env" alg.target_selection="$arg"
    done
done
