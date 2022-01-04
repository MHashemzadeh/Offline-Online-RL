#!/bin/bash
#SBATCH --job-name=acr_off_50k_tr
#SBATCH --output=../out/%A%a.out
#SBATCH --error=../out/%A%a.err

#SBATCH --array=0-0:1
#SBATCH --time=24:50:00
# SBATCH --account=def-whitem
#SBATCH --account=rrg-whitem

# SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4000M
# SBATCH --mem-per-cpu=64000M
#SBATCH --mail-user=m.hashemzadeh.b@gmail.com
#SBATCH --mail-type=FAIL

module load python
# source /home/maryam68/projects/def-afyshe-ab/maryam68/bin/activate
source ~/envt2/bin/activate

# DIR=/home/sungsu/workspace/actiongeneral/python


echo Running..$SLURM_ARRAY_TASK_ID

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+5"


#python create_data.py --offline_online_training 'offline' --mem_size 10000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15

#python create_data.py --offline_online_training 'offline' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15

#python create_data.py --offline_online_training 'offline' --mem_size 5000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15

#python create_data.py --offline_online_training 'offline' --mem_size 10000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6

#python create_data.py --offline_online_training 'offline' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6

#python create_data.py --offline_online_training 'offline' --mem_size 5000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6


python create_data.py --offline_online_training 'offline' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15 --tr_num_updates_pretrain 70

python create_data.py --offline_online_training 'offline' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15 --tr_num_updates_pretrain 150

python create_data.py --offline_online_training 'offline' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15 --tr_num_updates_pretrain 200

