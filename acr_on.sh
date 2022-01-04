#!/bin/bash
#SBATCH --job-name=seaquest
#SBATCH --output=../out/%A%a.out
#SBATCH --error=../out/%A%a.err

#SBATCH --array=8-23:1
#SBATCH --time=14:50:00
# SBATCH --account=def-whitem
#SBATCH --account=rrg-whitem

# SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10000M
# SBATCH --mem-per-cpu=64000M
#SBATCH --mail-user=m.hashemzadeh.b@gmail.com
#SBATCH --mail-type=FAIL

module load python
source ../minatar/bin/activate

# DIR=/home/sungsu/workspace/actiongeneral/python


echo Running..$SLURM_ARRAY_TASK_ID

#increment=1
#let "end_idx=$SLURM_ARRAY_TASK_ID+5"


python create_data.py --en 'seaquest' --tr_hyper_num $SLURM_ARRAY_TASK_ID

# seaquest, asterix , breakout , freeway  , space_invaders



#python new_b_ttn_offline.py --algo 'fqi' --hyper_num 10 --mem_size 10000 --num_rep 5 --tr_bayesian_optimism_factor $i

#for i in $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})
#do
#  echo Running..$i
#  python new_b_ttn_offonline.py --algo 'fqi' --hyper_num 10 --mem_size 10000 --num_rep 5 --tr_bayesian_optimism_factor $i
#
##  python new_b_ttn_offline.py --algo 'fqi' --hyper_num 10 --mem_size 10000 --num_rep 5 --tr_bayesian_optimism_factor $i
#
#done

#python create_data.py --offline_online_training 'online' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15

#python create_data.py --offline_online_training 'online' --mem_size 10000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6

#python create_data.py --offline_online_training 'online' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6



#python create_data.py --offline_online_training 'online' --mem_size 10000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15 --tr_initial_batch True

#python create_data.py --offline_online_training 'online' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15 --tr_initial_batch True

#python create_data.py --offline_online_training 'online' --mem_size 10000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6 --tr_initial_batch True

#python create_data.py --offline_online_training 'online' --mem_size 50000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6 --tr_initial_batch True



#python create_data.py --offline_online_training 'online' --mem_size 5000 --en 'Acrobot' --tr_alg_type 'fqi' --tr_hyper_num 15

#python create_data.py --offline_online_training 'online' --mem_size 5000 --en 'Acrobot' --tr_alg_type 'dqn' --tr_hyper_num 6





