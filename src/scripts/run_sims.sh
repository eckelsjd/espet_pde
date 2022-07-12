#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=run_sims
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load matlab/R2021b
matlab -nodisplay -r "run_sims ; exit"

echo "Finishing job script..."
