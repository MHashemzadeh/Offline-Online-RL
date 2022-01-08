#!/bin/bash
#SBATCH --job-name=seaquest_on
#SBATCH --output=../out/%A%a.out
#SBATCH --error=../out/%A%a.err
​
#SBATCH --array=0-10:1
#SBATCH --time=24:50:00
# SBATCH --account=def-whitem
#SBATCH --account=rrg-whitem
​
# SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8000M
# SBATCH --mem-per-cpu=64000M
#SBATCH --mail-user=m.hashemzadeh.b@gmail.com
#SBATCH --mail-type=FAIL
​
module load python
source ../minatar/bin/activate
​
# DIR=/home/sungsu/workspace/actiongeneral/python
​
​
echo Running..$SLURM_ARRAY_TASK_ID
​
#increment=1
#let "end_idx=$SLURM_ARRAY_TASK_ID+5"
​
​
python create_data.py --en 'seaquest' --tr_alg_type 'fqi' --tr_hyper_num SLURM_ARRAY_TASK_ID --offline_online_training 'offline' -tr_num_updates_pretrain 100