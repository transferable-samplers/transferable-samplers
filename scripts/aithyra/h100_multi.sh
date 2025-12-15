#!/bin/bash
#SBATCH --job-name=ts
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2            # Request 2 H100 NVL GPUs
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20              # 40 CPU cores (20 per GPU)
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00               # 3 days time limit
#SBATCH --output=logs/h100_job_%j.out
#SBATCH --error=logs/h100_job_%j.err

# IMPORTANT: Ensure your data is in Azure-local storage BEFORE submitting this job!
# See the "Data Transfer Strategy for Azure H100 Nodes" section

echo "==========================================="
echo "Job started: $(date)"
echo "Running on Azure node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="


# --- CONFIGURATION ---
# Path to your packed uv environment (created with venv-pack)
PACKED_ENV="/nfsdata/AITHYRA/shared/atong/ts-env.tar.gz"
# Where to unpack it (local scratch)
LOCAL_ENV="/mnt/localdata/scratch/env"
# Set paths - USE AZURE-LOCAL STORAGE
AZURE_DATA="/nfsdata/AITHYRA/shared/$USER"
export SCRATCH_DIR=$AZURE_DATA/scratch/transferable-samplers/

# Disables infiniband temporarily may be slower
#export NCCL_IB_DISABLE=1
# 1. Force the "Handshake" to happen over Ethernet (Reliable)
# On Azure, the control plane is usually eth0. Without this, it might try to handshake over IB.
export NCCL_SOCKET_IFNAME=eth0

# 2. Enable Detailed Logging
# If it fails again, this will tell us EXACTLY which interface it tried to pick.
export NCCL_DEBUG=INFO

# 3. Increase Memory Limits (Crucial for InfiniBand)
# 'ibv_create_qp' often fails if it can't lock enough memory.
ulimit -l unlimited

# Controls how long NCCL waits for the initial handshake between nodes
# Default is 15-30 minutes. Set to 60 seconds for debugging.
export NCCL_COMM_ID_TIMEOUT=60

# Controls how long PyTorch waits for other processes to join the group
# Useful if one node fails early and the other is just hanging.
export TORCH_NCCL_BLOCKING_WAIT=1  # Forces synchronous error reporting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# (Optional) Make PyTorch crash immediately on distributed errors with a stack trace
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# We override ntasks-per-node to 1 here so we don't unzip 4 times
srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --cpu-bind=none \
     bash -c "mkdir -p $LOCAL_ENV && tar -xf $PACKED_ENV -C $LOCAL_ENV"

#srun --nodes=$SLURM_JOB_NUM_NODES \
#     --ntasks=$SLURM_JOB_NUM_NODES \
#     --ntasks-per-node=1 \
#     --cpu-bind=none \
#echo "Setup complete. Environment ready at $LOCAL_ENV"

#srun --nodes=$SLURM_JOB_NUM_NODES \
#     --ntasks=$SLURM_JOB_NUM_NODES \
#     --ntasks-per-node=1 \
#     --cpu-bind=none \
#     $LOCAL_ENV/bin/python -c "import torch; print(torch.cuda.is_available())"

# Run your large-scale training
echo "Starting training on H100 GPUs..."
srun --cpu-bind=none \
	$LOCAL_ENV/bin/python /nfsdata/AITHYRA/shared/atong/transferable-samplers/src/train.py experiment=training/single_system/tarflow_Ace-A-Nme trainer=ddp data.batch_size=8192 trainer.num_nodes=2

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
