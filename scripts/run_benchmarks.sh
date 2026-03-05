#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark/%A_%a.out
#SBATCH --error=logs/benchmark/%A_%a.err
#SBATCH --array=0-19
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00

# 4 tests x 5 repeats = 20 tasks
# array index -> (test_idx, repeat_idx)
TESTS=(
    "tests/benchmark/test_snis_prose_up_to_8aa.py"
    "tests/benchmark/test_snis_ecnf_up_to_4aa.py"
    "tests/benchmark/test_smc_prose_up_to_8aa.py"
    "tests/benchmark/test_self_improve_prose_up_to_8aa.py"
)

NUM_REPEATS=5

TEST_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_REPEATS ))
REPEAT_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))
TEST_FILE=${TESTS[$TEST_IDX]}

echo "=== Job $SLURM_ARRAY_TASK_ID: test=$TEST_FILE repeat=$REPEAT_IDX ==="

mkdir -p logs/benchmark

PYTEST_TRAINER=gpu pytest -vv -s "$TEST_FILE" 2>&1
