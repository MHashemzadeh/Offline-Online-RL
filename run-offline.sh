#!/bin/bash
#SBATCH --job-name=mountain_car_offline
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-149:1
#SBATCH --time=12:00:00
#SBATCH --account=def-whitem
#SBATCH --cpus-per-task=10
#SBATCH --mem=16000M
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL


#DIR=/home/sungsu/workspace/actiongeneral/python

echo Running..$SLURM_ARRAY_TASK_ID
#increment=1
#let "end_idx=$SLURM_ARRAY_TASK_ID+5"

python -u create_data.py --tr_hyper_num $SLURM_ARRAY_TASK_ID --offline_online_training 'offline'