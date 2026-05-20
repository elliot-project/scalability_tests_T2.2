#!/bin/bash
#SBATCH --account=cin_staff
#SBATCH --error=%j.err
#SBATCH --output=%j.out
##SBATCH --partition=boost_usr_prod
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=tokenization
##SBATCH --qos=boost_qos_dbg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
##SBATCH --exclusive
#SBATCH --mem=30800MB


export NGC_API_KEY= #your NVIDIA key here
export CONTAINER_TAG=nvcr.io/nvidia/nemo:25.02 #container you want to build

export SINGULARITY_CACHEDIR="$SCRATCH/.singularity/cache"
export SINGULARITY_TMPDIR="$SCRATCH/.singularity/tmp"



module load singularity

#singularity pull docker://$CONTAINER_TAG

singularity build nemo_2502.sif docker://$CONTAINER_TAG
