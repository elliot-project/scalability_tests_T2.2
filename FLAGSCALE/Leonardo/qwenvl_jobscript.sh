#!/bin/bash
#SBATCH --account=Ellio_Elliott
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
##SBATCH --partition=lrd_all_serial
#SBATCH --job-name=flagscale
#SBATCH --qos=boost_qos_dbg
##SBATCH --qos=boost_qos_bprod
##SBATCH --time=00:29:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

module load cuda
#source prova/bin/activate
# export PYTHONPATH=/leonardo_scratch/large/userinternal/rscheda0/FlagScale:$PYTHONPATH
# export PYTHONPATH=/leonardo_scratch/large/userinternal/rscheda0/FlagScale/prova/lib/python3.11/site-packages:/leonardo_scratch/large/userinternal/rscheda0/FlagScale/prova/lib/python3.11/site-packages/flagscale/train:${PYTHONPATH}

source /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/mioenv/bin/activate
export PYTHONPATH=/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/mioenv/lib/python3.11/site-packages:/leonardo_scratch/large/userinternal/dbrandon/ELLIOT/mioenv/lib/python3.11/site-packages/flagscale/train:${PYTHONPATH}

export GPUS_PER_NODE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=6000
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
export BNB_CUDA_VERSION=121

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
echo MASTER_ADDR_IP = $MASTER_ADDR_iP

export NCCL_IB_SL=1
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

case $(( ${SLURM_LOCALID} )) in
0) export UCX_NET_DEVICES=mlx5_0:1 CUDA_VISIBLE_DEVICES=0 ;;
1) export UCX_NET_DEVICES=mlx5_1:1 CUDA_VISIBLE_DEVICES=1 ;;
2) export UCX_NET_DEVICES=mlx5_2:1 CUDA_VISIBLE_DEVICES=2 ;;
3) export UCX_NET_DEVICES=mlx5_3:1 CUDA_VISIBLE_DEVICES=3 ;;
esac


echo $SLURM_JOB_NODELIST

##export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

#srun flagscale train qwen3 --config ./examples/qwen3/conf/train.yaml
MASTER_PORT=9251
export MSC_CONFIG=/leonardo/home/userinternal/dbrandon/.config/multistorageclient/config.yaml

#Qwenv 2.5
srun torchrun --rdzv_backend c10d --nnodes 2 --node_rank 0 --nproc_per_node 4 --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT\
     --log_dir ./logs --tee 3 /leonardo_scratch/large/userinternal/rscheda0/FlagScale/flagscale/train/megatron/train_qwen2_5_vl.py\
     --num-workers 2 --calculate-per-token-loss --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1\
     --disable-bias-linear --use-flash-attn --use-distributed-optimizer --sequence-parallel --use-mcore-models --transformer-impl "transformer_engine"\
     --recompute-method uniform --recompute-granularity full --recompute-num-layers 1 --bf16 --attention-softmax-in-fp32 --log-interval 1\
     --tensorboard-log-interval 1 --log-throughput --wandb-project train_qwen2_5_vl_7b --wandb-exp-name train_qwen2_5_vl_7b --log-params-norm\
     --log-num-zeros-in-grad --tensorboard-dir ./train_qwen2_5_vl_7b/tensorboard --wandb-save-dir ./train_qwen2_5_vl_7b/wandb --save-interval 1000\
     --save ./train_qwen2_5_vl_7b/checkpoints --attention-backend unfused --add-qkv-bias --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944\
     --num-attention-heads 28 --num-query-groups 4 --seq-length 2048 --max-padding-length 2048 --enable-variable-seq-lengths\
     --max-position-embeddings 128000 --swiglu --normalization RMSNorm --norm-epsilon 1e-06 --init-method-std 0.02 --attention-dropout 0.0\
     --hidden-dropout 0.0 --clip-grad 1.0 --train-iters 62 --eval-iters 0 --micro-batch-size 1 --global-batch-size 16 --group-query-attention\
     --no-masked-softmax-fusion --untie-embeddings-and-output-weights --position-embedding-type mrope --rotary-percent 1.0 --rotary-base 1000000\
     --rotary-seq-len-interpolation-factor 1 --mrope-section 16 24 24 --seed 42 --weight-decay 0.1 --adam-beta1 0.9\
     --adam-beta2 0.999 --lr 1e-05 --min-lr 1e-06 --lr-warmup-iters 10 --lr-decay-style cosine --vocab-size 152064 --extra-vocab-size 421\
     --make-vocab-size-divisible-by 64 --no-gradient-accumulation-fusion --dataloader-type external\
     --tokenizer-path /leonardo_scratch/large/userinternal/rscheda0/FlagScale/Qwen/Qwen2.5-VL-7B-Instruct  --tokenizer-type Qwen2VLTokenizer \
     --disable-vision-class-token \
     --data-path /leonardo_scratch/large/userinternal/rscheda0/FlagScale/LLaVA-Pretrain/output/wds-1  \
     --vision-root /leonardo_scratch/large/userinternal/rscheda0/FlagScale/LLaVA-Pretrain      
     
     




## Qwen 3vl
# srun torchrun --rdzv_backend c10d --nnodes 2 --node_rank 0 --nproc_per_node 4 --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT\
#     --log_dir ./logs --tee 3 /leonardo_scratch/large/userinternal/rscheda0/FlagScale/flagscale/train/megatron/train_qwen3_vl.py\
#     --num-workers 2 --calculate-per-token-loss --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1\
#     --disable-bias-linear --use-flash-attn --use-distributed-optimizer --sequence-parallel --use-mcore-models --transformer-impl "transformer_engine"\
#     --recompute-method uniform --recompute-granularity full --recompute-num-layers 1 --bf16 --attention-softmax-in-fp32 --log-interval 1\
#     --tensorboard-log-interval 1 --log-throughput --wandb-project train_qwen2_5_vl_7b --wandb-exp-name train_qwen2_5_vl_7b --log-params-norm\
#     --log-num-zeros-in-grad --tensorboard-dir ./train_qwen2_5_vl_7b/tensorboard --wandb-save-dir ./train_qwen2_5_vl_7b/wandb --save-interval 1000\
#     --save ./train_qwen2_5_vl_7b/checkpoints --kv-channels 128 --qk-layernorm --attention-backend unfused --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288\
#     --num-attention-heads 32 --num-query-groups 8 --seq-length 2048 --max-padding-length 2048 --enable-variable-seq-lengths\
#     --max-position-embeddings 262144 --swiglu --normalization RMSNorm --norm-epsilon 1e-06 --init-method-std 0.02 --attention-dropout 0.0\
#     --hidden-dropout 0.0 --clip-grad 1.0 --train-iters 62 --eval-iters 0 --micro-batch-size 1 --global-batch-size 16 --group-query-attention\
#     --no-masked-softmax-fusion --untie-embeddings-and-output-weights --position-embedding-type mrope --rotary-percent 1.0 --rotary-base 5000000\
#     --rotary-seq-len-interpolation-factor 1 --mrope-section 24 20 20 --patch-size 16 --seed 42 --weight-decay 0.1 --adam-beta1 0.9\
#     --adam-beta2 0.999 --lr 1e-05 --min-lr 1e-06 --lr-warmup-iters 10 --lr-decay-style cosine --extra-vocab-size 293\
#     --make-vocab-size-divisible-by 64 --no-gradient-accumulation-fusion --dataloader-type external\
#     --tokenizer-path /leonardo_scratch/large/userinternal/rscheda0/FlagScale/Qwen3-VL-8B-Thinking  --tokenizer-type Qwen2VLTokenizer \
#     --disable-vision-class-token \
#     --data-path /leonardo_scratch/large/userinternal/rscheda0/FlagScale/LLaVA-Pretrain/output/wds-1 \
#     --vision-root /leonardo_scratch/large/userinternal/rscheda0/FlagScale/LLaVA-Pretrain \



