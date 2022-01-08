#!/bin/bash
#SBATCH --account=def-afyshe-ab
#SBATCH --job-name=offline-data-aug-rep-learn-0.1-0.8-1.4 # name of the job
#SBATCH --output=offline-data-aug-rep-learn-0.1-0.8-1.4.txt # the output file name.
#SBATCH --cpus-per-task=8
#SBATCH --mem=10000MB
#SBATCH --time=03:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15 
