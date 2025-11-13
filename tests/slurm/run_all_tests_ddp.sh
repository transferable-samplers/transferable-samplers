#!/bin/bash
#SBATCH -J pytest-ddp                 # Job name
#SBATCH -o tests/watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH --partition=short-unkillable
#SBATCH -t 1:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:l40s:4                  # Type/number of GPUs needed
#SBATCH -c 8
#SBATCH --get-user-env

export PYTEST_TRAINER="ddp" # this is default value but made explicit here

pytest tests -v -s --junitxml=tests/report_${PYTEST_TRAINER}.xml