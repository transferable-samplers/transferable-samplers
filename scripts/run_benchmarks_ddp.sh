#!/bin/bash
#SBATCH --job-name=benchmark-ddp
#SBATCH --output=logs/benchmark/%A_%a.out
#SBATCH --error=logs/benchmark/%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH --partition=short-unkillable

TESTS=(
    "tests/benchmark/test_snis_prose_up_to_8aa.py"
    "tests/benchmark/test_snis_ecnf_up_to_4aa.py"
    "tests/benchmark/test_smc_prose_up_to_8aa.py"
    "tests/benchmark/test_self_improve_prose_up_to_8aa.py"
)

TEST_FILE=${TESTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Job $SLURM_ARRAY_TASK_ID: test=$TEST_FILE (DDP) ==="

mkdir -p logs/benchmark

PYTEST_TRAINER=ddp pytest -vv -s "$TEST_FILE" 2>&1
