#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:59:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -A Ellio_Elliott
#SBATCH --job-name=fv-finetune
#SBATCH --partition=boost_usr_prod
## SBATCH --mail-type=all
## SBATCH --mail-user=r.scheda@cineca.it

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

# load env
#source /leonardo_scratch/large/userinternal/rscheda0/ELLIOT/env_torch/bin/activate
source /leonardo_scratch/large/userinternal/rscheda0/ELLIOT/scalability_tests_T2.2/VLM-TRAINING/Leonardo/.venv/bin/activate

module load cuda/12.6
module load gcc

sleep 5

which wandb

# export OMP_NUM_THREADS=16
# export MKL_NUM_THREADS=16
# export NCCL_P2P_LEVEL=NVL

# export LOGLEVEL=INFO

# # debugging flags (optional)
# export NCCL_DEBUG=WARN
# export PYTHONFAULTHANDLER=1
# export NCCL_NVLS_ENABLE=0
# # optional debug settings
# # export NCCL_DEBUG=INFO
# # NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# # on your cluster you might need these:
# # set the network interface
# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# export NCCL_BUFFSIZE=2097152
# #export TORCH_DIST_INIT_BARRIER=1
# export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

lscpu | grep "NUMA"
taskset -cp $$
ulimit -l unlimited
ulimit -s unlimited

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co



#export NCCL_NVLS_ENABLE=0



wandb enabled
wandb offline

# *****
NGPUS=4
NNODES=2
# *****

srun --cpu-bind=none torchrun --nproc_per_node=$NGPUS \
                --nnodes=$NNODES \
                --rdzv_id 101 \
                --rdzv_backend c10d \
                --rdzv_endpoint "$head_node_ip:29500" \
                -m train.train_qwen \
		--config /leonardo_scratch/large/userinternal/rscheda0/ELLIOT/vlm-training/configs/leonardo/leonardo_config_debug.toml \
