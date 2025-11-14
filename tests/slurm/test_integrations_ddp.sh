#!/bin/bash
#SBATCH -J pytest-ddp                 # Job name
#SBATCH -o tests/watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH --partition=short-unkillable
#SBATCH -t 1:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:l40s:4                  # Type/number of GPUs needed
#SBATCH -c 8
#SBATCH --get-user-env

export PYTEST_REPORT_DIR="tests/reports"
export PYTEST_TRAINER="ddp"

pytest tests -v -m "pipeline" --junitxml=${PYTEST_REPORT_DIR}/report_${PYTEST_TRAINER}.xml 