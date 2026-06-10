# VLM-Training on Leonardo


Load needed modules:
```
module load cuda/12.6
module load gcc
module load python
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.6/none
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Create the virtualenv:
```
python -m venv .venv
source .venv/bin/activate
```

Install requirements:

```
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 
```

Install Causal_conv1d:
```
export WORKDIR=$PWD
cd /tmp
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.0.post8
TORCH_CUDA_ARCH_LIST="8.0" pip install . --no-build-isolation
cd $WORKDIR
```


Install flash attention:

```
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu126torch2.10-cp311-cp311-linux_x86_64.whl
```

Clone the repo:
```
git clone https://github.com/VLR-CVC/vlm-training/
```

Copy leonardo scripts to the repo:
```
cp -r leonardo/ vlm-training/configs/
cp multinode_leonardo.sh vlm-training/scripts/
cp energon_dataloader.py vlm-training/data/
```

N.B. Change the line `20` of `vlm-training/models/qwen3_vl/model.py` with `HAS_FLASH = True`
```
try:
    from flash_attn import flash_attn_varlen_func
    logger.info('Using FLASH_ATTENTION from `flash_attn`')
    HAS_FLASH = True
except ImportError:
    logger.info('Using FLASH_ATTENTION from `torch.nn.attention.varlen`')
    HAS_FLASH = False
    
```


To download models and datasets, follow the instructions [here](https://github.com/VLR-CVC/vlm-training/blob/main/USAGE.md)


Run the experiment:
```
cd vlm-training
sbatch scripts/multinode_leonardo.sh
```

N.B. Be sure to add these lines in the jobscript:
```
export NCCL_NVLS_ENABLE=0
```

