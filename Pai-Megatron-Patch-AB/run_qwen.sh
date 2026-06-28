#!/bin/bash
#SBATCH --account=Ellio_Elliott
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=training-qwen3-vl
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --exclusive

#added to check if it fix the [rank0]: Failed to open libnvidia-ml.so.1. The module load only does not work, adding also the ld_preload variable export when launching the container
#module load cuda/12.6

# ── Paths (adjust these) ─────────────────────────
SIF=/leonardo_scratch/large/userinternal/dbrandon/qwen3vl_pai_megatron.sif
#OVERLAY=/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/over_AB10.img
TRAINING_SCRIPT=/workspace/Pai-Megatron-Patch/examples/qwen3_vl/run_mcore_qwen.sh
WDS_OUTPUT_DIR=/workspace/data
CHECKPOINT_PATH=/mnt/qwen3-vl-ckpts/Qwen3-VL-30B-A3B-Instruct
TORCH_DIR=/tmp/torch_extensions
MEGATRON_BUILD_DIR=/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron_fused_kernels_build
TOKENIZER_DIR=/mnt/data/qwen3_vl_tokenizer_only
OUTPUT_BASE=/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/scalability_tests_T2.2/output_mcore_qwen3vl_2b_custom
mkdir -p ${MEGATRON_BUILD_DIR}
mkdir -p ${OUTPUT_BASE}

# ── Distributed training env vars ─────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_NTASKS_PER_NODE

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

export APPTAINER_NV_PATH=/usr/lib64
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}


mkdir -p logs

BINDS="$CUDA_HOME,${OUTPUT_BASE}:${OUTPUT_BASE}, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/DATA/LLaVA-Pretrain/wds2:${WDS_OUTPUT_DIR}, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/models/Qwen3-VL-30B-A3B-Instruct:${CHECKPOINT_PATH},  /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/models/Qwen3-VL-30B-A3B-Instruct:${TOKENIZER_DIR}, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron_datasets:/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/megatron/core/datasets, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/torch_extensions:${TORCH_DIR}, ${MEGATRON_BUILD_DIR}:/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/megatron/legacy/fused_kernels/build, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/output_mcore_qwen3vl_2b_custom:/workspace/output_mcore_qwen3vl_2b_custom, /usr/lib64/libnvidia-ml.so.1:/usr/lib64/libnvidia-ml.so.1,/usr/lib64/libnvidia-ml.so.535.274.02:/usr/lib64/libnvidia-ml.so.535.274.02"

#BINDS="$CUDA_HOME,${OUTPUT_BASE}:${OUTPUT_BASE}, /leonardo_work/Ellio_Elliott/data/finevision_instruct_AB:${WDS_OUTPUT_DIR}, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/models/Qwen3-VL-30B-A3B-Instruct:${CHECKPOINT_PATH},  /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/models/Qwen3-VL-30B-A3B-Instruct:${TOKENIZER_DIR}, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/megatron_datasets:/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/megatron/core/datasets, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/torch_extensions:${TORCH_DIR}, ${MEGATRON_BUILD_DIR}:/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/megatron/legacy/fused_kernels/build, /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/output_mcore_qwen3vl_2b_custom:/workspace/output_mcore_qwen3vl_2b_custom, /usr/lib64/libnvidia-ml.so.1:/usr/lib64/libnvidia-ml.so.1,/usr/lib64/libnvidia-ml.so.535.274.02:/usr/lib64/libnvidia-ml.so.535.274.02"

################INPUT FOR THE SCRIPT
### BASE CONFIG ###
MODEL_SIZE=2B
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=4096
PAD_LEN=4096
PR=bf16
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=4
PP=1
CP=1
ETP=1 
EP=1
SP=1
DO=true
FL=true
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=none
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=500
DATASET_PATH=${WDS_OUTPUT_DIR}       # mmap prefix, no .bin/.idx
VALID_DATASET_PATH=${WDS_OUTPUT_DIR}
PRETRAIN_CHECKPOINT_PATH=/mnt/data/qwen3_vl_tokenizer_only
TRAIN_ITERS=10
LR_WARMUP_ITERS=2
EVAL_ITERS=0

OUTPUT_BASEPATH=/workspace/output_mcore_qwen3vl_2b_custom

srun apptainer exec \
    --nv \
    -B "$BINDS" \
    --env your_wds_output_dir=${WDS_OUTPUT_DIR} \
    --env PRETRAIN_CHECKPOINT_PATH=${CHECKPOINT_PATH} \
    --env LD_PRELOAD="" \
    ${SIF} \
    bash -c "
        export MASTER_ADDR=${MASTER_ADDR}
        export MASTER_PORT=${MASTER_PORT}
        export WORLD_SIZE=${WORLD_SIZE}
        export NNODES=${NNODES}
        export NPROC_PER_NODE=${NPROC_PER_NODE}
        export NODE_RANK=\${SLURM_NODEID}
        export LOCAL_RANK=\${SLURM_LOCALID}
        export RANK=\${SLURM_PROCID}
        export CUDNN_PATH=/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/pyenv/lib/python3.12/site-packages/nvidia/cudnn
        export NVTE_FRAMEWORK=pytorch
        export MP_SFT_PACKING=false
        export TORCH_EXTENSIONS_DIR=$TORCH_DIR
        export TRITON_LIBCUDA_PATH=/tmp/cuda_stubs
        export LD_LIBRARY_PATH=/usr/lib64:/tmp/cuda_stubs:/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
        cd /workspace/Pai-Megatron-Patch/examples/qwen3_vl
        bash ${TRAINING_SCRIPT} \
            dsw \
            ${MODEL_SIZE} \
            ${BATCH_SIZE} \
            ${GLOBAL_BATCH_SIZE} \
            ${LR} \
            ${MIN_LR} \
            ${SEQ_LEN} \
            ${PAD_LEN} \
            ${PR} \
            ${TP} \
            ${PP} \
            ${CP} \
            ${ETP} \
            ${EP} \
            ${SP} \
            ${DO} \
            ${FL} \
            ${AC} \
            ${OPTIMIZER_OFFLOAD} \
            ${SAVE_INTERVAL} \
            ${DATASET_PATH} \
            ${VALID_DATASET_PATH} \
            ${PRETRAIN_CHECKPOINT_PATH} \
            ${TRAIN_ITERS} \
            ${LR_WARMUP_ITERS} \
            ${OUTPUT_BASEPATH}
    "

