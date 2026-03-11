#!/usr/bin/env bash
#SBATCH --gres gpu:6
#SBATCH --mem 128GB
#SBATCH --cpus-per-task 12
#SBATCH --time 12:00:00
#SBATCH --constrain "eowyn|arwen|galadriel|balrog|smaug|shire|gondor" 
#SBATCH --output /Midgard/home/gawtam/crg-telograf/cluster_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/gawtam/crg-telograf/cluster_logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/gawtam/miniforge3/envs/diffusion/bin/python
ACCELERATE=/Midgard/home/gawtam/miniforge3/envs/diffusion/bin/accelerate
cd /Midgard/home/gawtam/crg-telograf/project/code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/gawtam/miniforge3

## For testing
# $ACCELERATE launch --num_processes 4 --multi_gpu  train.py \
#         --flow \
#         --first_sat_init \
#         --skip_first_eval \
#         --clip_max 50000 \
#         --max_sol_clip 2 \
#         --type_ratios 1 3 2 4 \
#         -T simple_gnn_F \
#         --num_evals 128 \
#         -b 1 \
#         --max_viz 128 \
#         --test_muls 1024 \
#         --val_eval_only

### For Training
$ACCELERATE launch --num_processes 6 --multi_gpu  train.py \
        --flow \
        --first_sat_init \
        --skip_first_eval \
        --clip_max 50000 \
        --max_sol_clip 2 \
        --type_ratios 1 3 2 4
## rsync 
#!/bin/bash

# LOCAL_DIR="./"
# REMOTE_DIR="/Midgard/home/gawtam/crg-telograf/"
# REMOTE_SSH="slurmcluster"

# alias rpull="rsync -a $REMOTE_SSH:$REMOTE_DIR/ $LOCAL_DIR/"
# alias rpush="rsync -a --delete $LOCAL_DIR/ $REMOTE_SSH:$REMOTE_DIR/"