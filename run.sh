#!/bin/bash
#SBATCH --account=def-afyshe-ab
#SBATCH --job-name=offline-existing-data-augmentation-ras_alpha-0.6-ras_beta-1.4 # name of the job
#SBATCH --output=offline-existing-data-augmentation-ras_alpha-0.6-ras_beta-1.4.txt # the output file name.
#SBATCH --cpus-per-task=8
#SBATCH --mem=10000MB
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15 
