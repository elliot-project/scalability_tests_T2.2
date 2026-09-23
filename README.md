# scalability_tests_T2.2

Repository collecting scripts, notes, and experiment assets used to run **scalability and training tests for vision-language models (VLMs)** across different software stacks and HPC environments.

## Results

Results doc: https://docs.google.com/spreadsheets/d/1LQnzmETiSy2s-e-xd4CXB_8pV_L6EVy2dX2mZlQUgvI/edit?gid=0#gid=0

## Repository purpose

This repository is organized as a workspace for comparing and reproducing VLM training experiments with different frameworks:

- **FlagScale**-based runs
- **Megatron / NeMo**-based runs
- **Pai-Megatron-Patch** experiments for Qwen3-VL
- **VLM-Training** experiments
- **TorchTitan**-based runs

The material currently focuses on **Leonardo**, **Lumi**, **MN5** and **JUPITER** cluster runs.

## Repository structure

```text
scalability_tests_T2.2/
├── FLAGSCALE/
├── MEGATRON/
├── Pai-Megatron-Patch-AB/
├── Torchtitan/
├── VLM-TRAINING/
└── README.md
```

## Directory overview

### `FLAGSCALE/`
Assets for running VLM experiments with the **FlagScale** stack.

Current content includes:

- `Leonardo/README.md` with setup notes for installing FlagScale dependencies
- `Leonardo/qwenvl_jobscript.sh` with a multi-node SLURM launch script
- `Leonardo/setup_venv.sh` for environment setup
- `Leonardo/modified_dataset_helpers.py` for dataset-related adjustments
- `Lumi/` and `MN5/` directories reserved for cluster-specific material

This area is useful if you want to reproduce or adapt experiments based on the FlagScale training flow.

### `MEGATRON/`
Experiments and instructions based on **Megatron-LM / NVIDIA NeMo** for VLM training.

Current content includes:

- `Leonardo/README.md` with instructions for:
  - creating a Singularity container from `nvcr.io/nvidia/nemo:25.02`
  - preparing the LLaVA-Pretrain dataset
  - converting data to WebDataset and Megatron-Energon format
  - launching Qwen2.5-VL training
- `Leonardo/create_container.sh` to build the NeMo container
- `Leonardo/train_qwen2_5-vl.sh` and `Leonardo/train_qwen2_5-vl_llava.sh` for training jobs
- `Leonardo/ft_config.yaml` for configuration
- `Leonardo/nvrx_qwen2vl.sh` for additional launch/runtime support
- `Lumi/` and `MN5/` directories reserved for future platform-specific variants

This section is the reference point for Megatron-based scalability tests in the repository.

### `Pai-Megatron-Patch-AB/`
Experiment material for **Qwen3-VL training with Pai-Megatron-Patch**.

Current content includes:

- `README.md` describing:
  - container creation with Apptainer
  - fake WebDataset creation
  - metadata generation
  - training launch with `sbatch run_qwen.sh`
- `qwen3vl_pai_megatron.def` to build the container image
- `run_qwen.sh` as the SLURM training launcher
- `48015044.out` and `48015044.err` as example job logs

Use this directory when working specifically with the Alibaba Pai-Megatron-Patch workflow.

### `Torchtitan/`
Assets for benchmarking **[TorchTitan](https://github.com/pytorch/torchtitan)**,
pinned at commit `b21f7d43e`.

Current content includes:

- `JUPITER/README.md` with the pinned revision, environment layout, the launch
  pitfall that makes the `torchrun` console script unusable, the mandatory
  `--training.disable-cuda-graphs`, and how to read the resulting numbers
- `JUPITER/config_registry.py` with the benchmark-only config functions, reached
  through `--module ttbench` and kept outside the pinned checkout
- `JUPITER/tt_bench.sbatch` and `JUPITER/submit_tt_sweep.sh` as the SLURM launchers
- `JUPITER/patches/attn_gym_gdn_fake_strides.py`, required on sm90
- `JUPITER/chat_template_sft.jinja`

JUPITER is the one cluster where two codebases were benchmarked against each
other, so this directory is the counterpart of `VLM-TRAINING/JUPITER/` — same
machine, same model, same dataset.

### `VLM-TRAINING/`
Material for experiments based on the **VLR-CVC `vlm-training`** project.

Current content includes:

- `Leonardo/README.md` with instructions for:
  - module loading
  - virtual environment creation
  - PyTorch installation
  - `causal-conv1d` and flash-attention setup
  - cloning and patching the upstream `vlm-training` repository
  - running distributed experiments on Leonardo
- `Leonardo/multinode_leonardo.sh` with the SLURM multi-node launch script
- `Leonardo/energon_dataloader.py` for data loading support
- `Leonardo/requirements.txt` for Python dependencies
- `JUPITER/README.md` with the JUPITER branch and revision, environment setup,
  and the cluster-specific constraints (no outbound network on compute nodes,
  `--gpus-per-task` rejected, node-local compile caches)
- `JUPITER/multinode_jup.sh` with the SLURM multi-node launch script
- `JUPITER/jupiter/` with the Qwen3-VL-2B and Qwen3.5-9B scaling configs
- `JUPITER/requirements.txt` for Python dependencies
- `LUMI/` with throughput plots per model

Note that `Leonardo/` and `JUPITER/` document **different revisions** of
`vlm-training` and their configs are not interchangeable; each README states the
branch it targets.




## Related per-directory documentation

- `Pai-Megatron-Patch-AB/README.md`
- `FLAGSCALE/Leonardo/README.md`
- `MEGATRON/Leonardo/README.md`
- `VLM-TRAINING/Leonardo/README.md`
- `VLM-TRAINING/JUPITER/README.md`
- `Torchtitan/JUPITER/README.md`
