#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=gen_data
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load matlab/R2021b
matlab -nodisplay -r "gen_data ; exit"

echo "Finishing job script..."
