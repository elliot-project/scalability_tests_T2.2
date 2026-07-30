#!/bin/bash
#SBATCH --account=cin_staff
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=build-container
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mem=30800MB

 export APPTAINER_CACHEDIR=$CINECA_SCRATCH
 export APPTAINER_TMPDIR=$CINECA_SCRATCH


 srun apptainer build --fakeroot megatron-bridge.sif build_container.def 
