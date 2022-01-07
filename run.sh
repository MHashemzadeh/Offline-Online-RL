#!/bin/bash
<<<<<<< HEAD
#SBATCH --account=def-afyshe-ab
#SBATCH --job-name=offline-existing-data-augmentation-ras_alpha-0.6-ras_beta-1.4 # name of the job
#SBATCH --output=offline-existing-data-augmentation-ras_alpha-0.6-ras_beta-1.4.txt # the output file name.
#SBATCH --cpus-per-task=8
=======
#SBATCH --account=def-whitem
#SBATCH --job-name=offline # name of the job
#SBATCH --output=offline.txt # the output file name.
#SBATCH --cpus-per-task=4
>>>>>>> 6b09cf6ab4a3c020dc1d69748e668f2f1a00bd2c
#SBATCH --mem=10000MB
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

<<<<<<< HEAD
python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15 
=======

python -u create_data.py --hyper_num 15 --mem_size 1000 --num_step_ratio_mem 50000 --en 'Mountaincar'
#python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15
>>>>>>> 6b09cf6ab4a3c020dc1d69748e668f2f1a00bd2c
