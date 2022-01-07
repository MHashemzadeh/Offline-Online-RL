#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --job-name=offline # name of the job
#SBATCH --output=offline.txt # the output file name.
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000MB
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL


python -u create_data.py --hyper_num 15 --mem_size 1000 --num_step_ratio_mem 50000 --en 'Mountaincar'
#python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15
