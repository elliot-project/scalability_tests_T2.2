# Megatron-Bridge on LUMI

This benchmark measures training throughput (TFLOP/s per GPU, tokens/s) for
`Qwen/Qwen3-VL-8B-Instruct` on LUMI (AMD MI250X, ROCm 7.0), using a LUMI/ROCm port
of NVIDIA's Megatron-Bridge. The code and the authoritative, up-to-date instructions
live in a standalone repository:

**https://github.com/gnaithani-csc/Megatron-Bridge-LUMI**  (see its `README_LUMI.md`)

The steps below are a fuller walkthrough of that repo for a first-time user.
**If anything here disagrees with `README_LUMI.md` in the repo, the repo is correct.**

---

## What you will do

1. Clone the repo and pull its `megatron-core` submodule.
2. Build a small Python venv on top of the LUMI container.
3. generate the synthetic dataset (for synthetic data instead of using the mock data settings).
4. Launch the benchmark, on one node interactively or several nodes via Slurm.

There are two run modes. The **mock** recipe feeds the model random tensors and needs
no data at all. The **energon** recipe trains on a synthetic caption dataset and needs step 3.

---

## 1. Get the code

The runtime `megatron-core` is referenced as a git submodule pinned to `core_v0.16.0`, so you pull it after cloning:

```bash
git clone https://github.com/gnaithani-csc/Megatron-Bridge-LUMI.git
cd Megatron-Bridge-LUMI
git submodule update --init 3rdparty/Megatron-LM
```

The port itself lives under `src/megatron/bridge/`. The launch scripts add both
`src/` and `3rdparty/Megatron-LM` to `PYTHONPATH` for you, so you do not install this
repo with pip.

## 2. Build the environment

A Singularity container provides the base stack (PyTorch for ROCm, Transformer Engine, apex, flash-attn), and you add a
small venv on top for the few extra packages this port needs.

First load the module that exports the container image path (`$SIF_plus`) and the
project directory (`$PROJ_DIR`):

```bash
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

`$SIF_plus` resolves to the container image used for these runs:
`/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif`

`/appl/local/laifs/containers` is the LUMI AI Factory Software (LAIFS) container
collection: a family of images built hierarchically, each adding functionality on top
of the previous. For the naming scheme, see the
[LAIFS container recipes releases](https://github.com/lumi-ai-factory/laifs-container-recipes/releases)
and the [LUMI AI software environment docs](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).

Then open a shell inside the container and create the venv. The
`--system-site-packages` flag is important: it lets the venv see the container's
torch, transformer-engine, and so on, so you only install what is missing.

```bash
singularity shell $SIF_plus
python -m venv env_megatron --system-site-packages
source env_megatron/bin/activate
pip install --no-build-isolation \
    megatron-energon==7.3.2 \
    transformers==5.5.4 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0
```

Notes:
- Build the venv on the **same container** you will run with. A venv built against a
  different container can pull mismatched packages that then can fail to import at runtime.
 
## 3. Prepare the dataset (energon recipe)

The synthetic dataset comes from a separate generator repository,
`elliot-project/synth-data-bench-training` (pinned at commit `cf96419`).

```bash
git clone https://github.com/elliot-project/synth-data-bench-training
cd synth-data-bench-training && git checkout cf96419
```

Back in the Megatron-Bridge-LUMI repo, one script runs the whole pipeline (generate,
convert to the encoder's format, then `energon prepare`) inside the container and venv:

```bash
bash scripts/data/prepare_synth_data.sh
```

Paths are configurable through environment variables documented in the script header
(`GENERATOR_REPO`, `GEN_CONFIG`, `GEN`, `OUT`). When it finishes it prints the
`dataset.path` to point the recipe at.

## 4. Run the benchmark

Both launch scripts wrap the same entrypoint:

```
torchrun ... scripts/training/run_recipe.py \
    --recipe <RECIPE> --step_func vlm_step --mode pretrain \
    --seq_length <N> <key=value overrides...>
```

`--step_func vlm_step` is the generic vision-language forward step. Overrides are
dot-path `key=value` pairs applied after the recipe is built (for example
`model.tensor_model_parallel_size=2`).

### Single node (interactive, 8 GPUs)

From inside a container shell on an allocated node, edit the `RECIPE` and
`CLI_OVERRIDES` at the top of the script if needed, then:

```bash
bash scripts/training/launch_interactive.sh
```

Defaults: the energon caption recipe, tensor-parallel size 2, global batch 64,
sequence length 8192, Transformer Engine kernels. Output is tee'd to `train.log`.

### Multiple nodes (Slurm)

Edit the `#SBATCH` header (`--nodes`, `--account`, `--partition`) and the config, then
submit from the repo root:

```bash
sbatch scripts/training/launch_with_sbatch.sh
```

Defaults: 4 nodes, tensor-parallel size 2, global batch 256, sequence length 8192.
Logs land under `logs/train_%j.out` and `logs/train_%j.err`.

**Scaling note:** for weak scaling, the global batch should grow with the GPU count so each GPU keeps
the same amount of work. 

### Recipes available

- `qwen3_vl_8b_throughput_mock_config` — random tensors, no data.
- `qwen3_vl_8b_synth_cap_energon_config` — synthetic captions (needs step 3), default.

They live in `src/megatron/bridge/recipes/qwen_vl/qwen3_vl.py`.

## 5. Results

Each logged step prints a line like:

```
iteration 10/10 | elapsed time per iteration (ms): xyz | throughput per GPU (TFLOP/s/GPU): xyz | ... | lm loss: ...
```


Full further details and the exact config settings, please refer `README_LUMI.md` in the repo.
