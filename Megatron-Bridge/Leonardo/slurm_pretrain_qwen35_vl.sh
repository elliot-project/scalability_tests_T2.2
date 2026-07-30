#!/bin/bash
#SBATCH --account=cin_staff
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=conver-ckpt
#SBATCH --time=00:10:00
##SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=5
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1



module load gcc
module load cuda/12.6


export LD_PRELOAD=""

export GPUS_PER_NODE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=6000
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo hostname = `hostname`
echo HOSTNAMES = $HOSTNAMES
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_PROCID= $SLURM_PROCID
echo NNODES= $NNODES
echo WORLD_SIZE= $WORLD_SIZE
echo NODE_RANK= $NODE_RANK
echo NODE_NAME = $SLURMD_NODENAME
echo MASTER_ADDR_IP = $MASTER_ADDR_IP

export NCCL_IB_SL=1
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_ALGO=^NVLS
export TOKENIZERS_PARALLELISM=false

case $(( ${SLURM_LOCALID} )) in
0) export UCX_NET_DEVICES=mlx5_0:1 CUDA_VISIBLE_DEVICES=0 ;;
1) export UCX_NET_DEVICES=mlx5_1:1 CUDA_VISIBLE_DEVICES=1 ;;
2) export UCX_NET_DEVICES=mlx5_2:1 CUDA_VISIBLE_DEVICES=2 ;;
3) export UCX_NET_DEVICES=mlx5_3:1 CUDA_VISIBLE_DEVICES=3 ;;
esac

echo $SLURM_JOB_NODELIST

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export WANDB_MODE=offline

# Workspace directory for checkpoints and results.
WORKSPACE=${WORKSPACE:-/workspace}
MODEL_PATH=${MODEL_PATH:-/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/models/Qwen3.5-27B}
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/prova/megatron-bridge/qwen35_vl_mimo}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_ROOT}/logs/mimo_pretrain}"
CKPT_DIR="${CKPT_DIR:-${EXPERIMENT_ROOT}/results/mimo_pretrain}"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" || { echo "ERROR: cannot create ${LOG_DIR}" >&2; exit 1; }
[ -w "${LOG_DIR}" ] || { echo "ERROR: ${LOG_DIR} is not writable" >&2; exit 1; }

MASTER_PORT=9251


export TRANSFORMERS_OFFLINE=1
SEQ_LENGTH=${SEQ_LENGTH:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
TRAIN_ITERS=${TRAIN_ITERS:-5}


export DISTRIBUTED_ARGS="--rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE}"

cmd=(torchrun $DISTRIBUTED_ARGS
     ./pretrain_qwen35_vl.py 
    --hf-model /workspace/model 
    --component "language=tp=4,pp=2,cp=1,dp=2,rank_offset=0" 
    --component "images=tp=1,pp=1,cp=1,dp=4,rank_offset=16" 
    --seq-length $SEQ_LENGTH 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE 
    --train-iters $TRAIN_ITERS 
    --lr-warmup-iters 5 
    --data-mode mock
    --experiment-root "/workspace/qwen35_vl_mimo"
    --log-dir "/workspace/qwen35_vl_mimo/logs/mimo_pretrain"
    --checkpoint-dir "/workspace/qwen35_vl_mimo/results/mimo_pretrain"
)

if [ -n "${NUM_MOE_EXPERTS}" ]; then
    cmd+=(--num-moe-experts "${NUM_MOE_EXPERTS}")
fi

if [ -n "${LOG_THROUGHPUT}" ]; then
    cmd+=(--log-throughput --throughput-window-size "${THROUGHPUT_WINDOW_SIZE}")
fi

if [ -n "${TENSORBOARD_DIR}" ]; then
    cmd+=(--tensorboard-dir "${TENSORBOARD_DIR}")
fi

CONTAINER=${CONTAINER:-/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/megatron-bridge.sif}

#BINDS="$CUDA_HOME,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/prova/megatron-bridge2/megatron-bridge/:/workspace/Megatron-Bridge,$TOKENIZER_PATH:/workspace/tokenizer/,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/tensorboard:/workspace/tensorboard, "
#BINDS="$CUDA_HOME,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/prova/megatron-bridge/:/workspace/Megatron-Bridge,$MODEL_PATH:/workspace/model/,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/tensorboard:/workspace/tensorboard,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/prova/megatron-bridge/qwen35_vl_mimo:/workspace/qwen35_vl_mimo"
BINDS="$CUDA_HOME,$MODEL_PATH:/workspace/model/,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/tensorboard:/workspace/tensorboard,/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron-bridge/prova/megatron-bridge/qwen35_vl_mimo:/workspace/qwen35_vl_mimo"


srun -l singularity exec --nv \
    -B "$BINDS" \
    $CONTAINER \
    "${cmd[@]}"

