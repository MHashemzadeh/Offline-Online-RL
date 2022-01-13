#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --job-name=online_hyper-num_102 # name of the job
#SBATCH --output=output_log/%A%a.out # the output file name.
#SBATCH --error=output_log/%A%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=10000MB
#SBATCH --time=03:00:00
#SBATCH --mail-user=awahab.bscs16seecs@seecs.edu.pk
#SBATCH --mail-type=ALL

python -u create_data.py --offline_online_training 'online' --tr_hyper_num 102 
