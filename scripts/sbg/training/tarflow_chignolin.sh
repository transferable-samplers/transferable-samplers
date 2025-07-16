#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=256G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4                  # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

env=tbg3
module purge
module load python/3.11 cuda/12.2
module load openmm/8.2.0
module load httpproxy/1.0
source $HOME/envs/$env/bin/activate
wandb online

RUN_NAME=tarflow_chignolin

srun python -u src/train.py \
experiment=training/single_system/tarflow_chignolin \
model.net.channels=384 \
logger=wandb \
trainer=ddp \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
