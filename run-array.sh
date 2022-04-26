#!/bin/bash
#SBATCH --job-name=dqn-smstde-CP-offline_online
#SBATCH --output=output_log/%A%a.out
#SBATCH --error=output_log/%A%a.err
#SBATCH --array=0-14:1
#SBATCH --time=12:00:00
#SBATCH --account=def-whitem
#SBATCH --cpus-per-task=10
#SBATCH --mem=16000M
#SBATCH --mail-user=kamranejaz98@gmail.com
#SBATCH --mail-type=ALL

echo Running..$SLURM_ARRAY_TASK_ID

module load python/3.8
python -u create_data.py --tr_hyper_num $SLURM_ARRAY_TASK_ID --offline_online_training 'offline_online'