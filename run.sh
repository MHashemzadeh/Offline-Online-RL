#!/bin/bash
#SBATCH --account=def-afyshe-ab
#SBATCH --job-name=offline-data-aug-rep-learn-0.1-0.8-1.4 # name of the job
#SBATCH --output=offline-data-aug-rep-learn-0.1-0.8-1.4.txt # the output file name.
#SBATCH --error=offline-data-aug-rep-learn-0.1-0.8-1.4-error.err
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10000MB
#SBATCH --mem-per-cpu=64000M
#SBATCH --time=03:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

python -u create_data.py --offline_online_training 'offline' --tr_hyper_num 15 
