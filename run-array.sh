#!/bin/bash
#SBATCH --job-name=sparsity_mountain_car_offline_online
#SBATCH --output=output_logs/%A%a.out
#SBATCH --error=output_logs/%A%a.err
#SBATCH --array=0-929:1
#SBATCH --time=12:00:00
#SBATCH --account=def-whitem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16000M
#SBATCH --mail-user=kamranejaz98@gmail.com
#SBATCH --mail-type=ALL


#DIR=/home/sungsu/workspace/actiongeneral/python

echo Running..$SLURM_ARRAY_TASK_ID
#increment=1
#let "end_idx=$SLURM_ARRAY_TASK_ID+5"

python -u create_data.py --tr_hyper_num $SLURM_ARRAY_TASK_ID --offline_online_training 'offline_online'
