#!/bin/bash
#SBATCH --account=Ellio_Elliott
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
##SBATCH --partition=lrd_all_serial
#SBATCH --job-name=megatron-qwenvl
#SBATCH --qos=boost_qos_dbg
##SBATCH --qos=boost_qos_bprod
##SBATCH --time=00:29:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

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

MASTER_PORT=9251

CONTAINER=/leonardo_scratch/large/userinternal/rscheda0/ELLIOT/megatron/nemo_2502.sif
#OVERLAY_PATH=/leonardo_scratch/large/userinternal/rscheda0/prova_flagscale/myover2.img



# Output directories
OUTPUT_BASE=/leonardo_scratch/large/userinternal/rscheda0/train_megatron_qwen2_5_vl_7b
TENSORBOARD_DIR=${OUTPUT_BASE}/tensorboard
CHECKPOINT_DIR=${OUTPUT_BASE}/checkpoints
LOGS_DIR=${OUTPUT_BASE}/logs
mkdir -p ${TENSORBOARD_DIR} ${CHECKPOINT_DIR} ${LOGS_DIR}

# Data: energon-format dataset YAML
# NOTE: Megatron-LM uses the energon dataloader. Create a dataset.yaml in energon
# format pointing to your wds shards, or set DATA_PATH to an existing energon dataset.
# See: /opt/megatron-lm/examples/multimodal/pretrain_dataset.yaml for the template.
# The LLaVA-Pretrain host dir is bound to /data/LLaVA-Pretrain inside the container.

DATA_PATH=/data/FlagScale/LLaVA-Pretrain/wds/wds
#DATA_PATH=/data/synth-data-bench-training/data/vqa

# Tokenizer and model paths
TOKENIZER_PATH=/leonardo_scratch/large/userinternal/rscheda0/FlagScale/Qwen/Qwen2.5-VL-7B-Instruct

# Parallelism
TP=4
PP=4
CP=1

# Sequence lengths
SEQ_LEN=256          # vision token sequence length (per image with internvit + pixel-shuffle)
DECODER_SEQ_LEN=2048 # language model context length (must be > max_num_tiles+1 * tokens_per_tile = 13*256=3328)
MAX_POS_EMBED=128000

# Batch sizes
MBZ=1
GBZ=1
NW=1

# === Distributed args ===
export DISTRIBUTED_ARGS="--rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE}"

# === Model args ===
export GPT_ARGS="\
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --use-distributed-optimizer \
    --language-model-type qwen2.5_7B \
    --vision-model-type internvit \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --num-query-groups 4 \
    --group-query-attention \
    --add-qkv-bias \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${MAX_POS_EMBED} \
    --img-h 448 \
    --img-w 448 \
    --patch-dim 14 \
    --use-tiling \
    --max-num-tiles 6 \
    --use-thumbnail \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${GBZ} \
    --calculate-per-token-loss \
    --bf16 \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 28 \
    --sequence-parallel \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-te \
    --pixel-shuffle"
# === Training args ===
export TRAIN_ARGS="\
    --train-iters 62 \
    --lr 1e-05 \
    --min-lr 1e-06 \
    --lr-warmup-iters 10 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --seed 42 \
    --init-method-std 0.02 \
    --eod-mask-loss"

# === Data args ===
export DATA_ARGS="\
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --tokenizer-prompt-format qwen2p5 \
    --data-path ${DATA_PATH} \
    --dataloader-type external \
    --prompt-path /opt/megatron-lm/examples/multimodal/nvlm/nvlm_prompts.json \
    --split 10,0,90 \
    --num-workers ${NW} \
    --eval-iters 0"

# === Output / logging args ===
export OUTPUT_ARGS="\
    --log-interval 1 \
    --tensorboard-log-interval 1 \
    --log-throughput \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 1000 \
    --save ${CHECKPOINT_DIR}"

export LOGGING_ARGS="\
    --wandb-project train_megatron_qwen2_5_vl_7b \
    --wandb-exp-name train_megatron_qwen2_5_vl_7b"

# === Bind mounts ===
BINDS="$CUDA_HOME,${OUTPUT_BASE}:${OUTPUT_BASE},/leonardo_scratch/large/userinternal/rscheda0/:/data/,${TOKENIZER_PATH}:${TOKENIZER_PATH}"

# === Launch ===
srun -l singularity exec --nv \
    -B "$BINDS" \
    $CONTAINER \
    torchrun $DISTRIBUTED_ARGS \
    /opt/megatron-lm/examples/multimodal/train.py \
    $GPT_ARGS $TRAIN_ARGS $DATA_ARGS $OUTPUT_ARGS $LOGGING_ARGS
    #--transformer-impl transformer_engine \
    
