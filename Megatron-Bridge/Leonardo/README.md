# Qwen3.5-VL Pretraining Script (Megatron-Bridge)

This repository contains the implementation of  Qwen3.5-VL **pretraining**, extending the finetuning example provided by NVIDIA's Megatron-Bridge repository.

## Origin

This script builds on the finetuning example found here:

- [Megatron-Bridge/examples/megatron_mimo/qwen35_vl](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/megatron_mimo/qwen35_vl)

## How to use

This script is **not standalone** — it depends on the Megatron-Bridge codebase and must be placed inside it to run.

1. Clone Megatron-Bridge:
```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
   cd Megatron-Bridge
```

2. Copy `pretraining_qwen.py` from this repo into the target directory:
```bash
   cp /path/to/this-repo/pretraining_qwen.py examples/megatron_mimo/qwen35_vl/
   cp /path/to/this-repo/slurm_pretrain_qwen35_vl.sh examples/megatron_mimo/qwen35_vl/
```

3. Build the container using build_container.sh 

4. Change the following paths 
```WORKSPACE
MODEL_PATH
EXPERIMENT_ROOT
LOG_DIR
CKPT_DIR
```
in the slurm_pretrain_qwen35_vl.sh script 

5. Run pretraining from within `examples/megatron_mimo/qwen35_vl/`:
```sbatch slurm_pretrain_qwen35_vl.sh
```

## Notes

- This file must live at `Megatron-Bridge/examples/megatron_mimo/qwen35_vl/pretraining_qwen.py` to resolve its relative imports/config paths correctly.
- Any updates to the upstream Megatron-Bridge repo (especially in `examples/megatron_mimo/qwen35_vl`) may require adjusting this script accordingly.
