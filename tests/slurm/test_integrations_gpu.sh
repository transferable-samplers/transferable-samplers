#!/bin/bash
#SBATCH -J pytest-gpu                 # Job name
#SBATCH -o tests/watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH --partition=main,long
#SBATCH -t 1:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:l40s:1                  # Type/number of GPUs needed
#SBATCH -c 8
#SBATCH --get-user-env

export PYTEST_REPORT_DIR="tests/reports"
export PYTEST_TRAINER="gpu" # this is default value but made explicit here

pytest tests -v  -m "pipeline" --junitxml=${PYTEST_REPORT_DIR}/report_${PYTEST_TRAINER}.xml
