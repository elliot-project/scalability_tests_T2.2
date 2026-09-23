# VLM-Training on JUPITER

JUPITER (JSC): GH200 nodes, **4 GPUs per node**, `booster` partition, aarch64.

## Which revision

These configs target the **`titan-migration`** branch of
[VLR-CVC/vlm-training](https://github.com/VLR-CVC/vlm-training), at `de51baa`.

```bash
git clone https://github.com/VLR-CVC/vlm-training/
cd vlm-training
git checkout titan-migration
```

> **They will not parse against `main`.** The branch renamed and removed config
> fields — `model_impl` and `data.batch_size`, which `VLM-TRAINING/Leonardo`
> still uses, no longer exist, and `model.model_config` (a path to an HF-format
> `config.json`) is now what selects the architecture. The two clusters'
> configs document different revisions and are not interchangeable.

## Environment

JUPITER already ships a conda env with a working aarch64 CUDA torch; the repo
uses it rather than building one:

```bash
source /e/project1/open-sci-mm/ockier1/envs/miniforge3/etc/profile.d/conda.sh
conda activate /e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main
module load CUDA/13
```

`requirements.txt` is the repo's shared dependency set, copied here for
reference. It is not JUPITER-specific and is already satisfied inside
`torch_main`; do not `pip install` into that env, it is shared.

## Install the launch assets

```bash
cp -r jupiter/ vlm-training/configs/
cp multinode_jup.sh vlm-training/scripts/
```

Model weights and datasets: see
[USAGE.md](https://github.com/VLR-CVC/vlm-training/blob/main/USAGE.md).

## Run

```bash
cd vlm-training
sbatch --nodes=<N> scripts/multinode_jup.sh
```

Node count comes from `sbatch --nodes`, so one script serves every point of a
scaling sweep. Edit `CONFIG_FILE` inside the script to pick the model.

## What is JUPITER-specific

**No outbound network on compute nodes**, which is why the job sets:

```bash
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co
```

wandb runs are written offline and uploaded afterwards from a machine that has
network.

**`--gpus-per-task` is rejected** by JUPITER's Slurm with `Invalid GRES
specification`. The job uses `--ntasks-per-node=1` with `--gpus-per-node=4`,
which is the same request.

**Compile caches must be node-local.** Booster nodes are diskless: `/` is an
overlay whose writable layer is a ~96 GiB RAM tmpfs. With a shared GPFS inductor
cache, ~1,100 nodes starting at once stalled in their first step, ranks waiting
past the NCCL timeouts on another node's file locks. Every job compiles cold
instead.

**Threads:** `OMP_NUM_THREADS=64`, `OPENBLAS_NUM_THREADS=64` (288 cores/node).

## Configs

| file | model | seq_len | TP | notes |
|---|---|---|---|---|
| `jupiter/qwen3_vl_2b.toml` | Qwen3-VL-2B | 32768 | 2 | the scaling-sweep workload |
| `jupiter/qwen3_5_9b.toml` | Qwen3.5-9B | 16384 | 4 | TP 4 keeps tensor parallelism inside a node |

Both use `data_parallel = "fsdp"` and `reshard_after_forward = "never"`, so
adding nodes adds data parallelism only — that is the axis the sweeps measure.
`save_steps` is set above `total_steps` so a benchmark never checkpoints
mid-run.

## Results

All figures in `plots/`. Qwen3-VL-2B, plotqa_cot, `seq_len = 32768`, `tp=2`,
fp32 master weights, `reshard_after_forward = "never"`.

### `qwen3_vl_2b_weak_vs_strong.png`
Solid lines are full-rate steps; dashed is the post-warmup mean, which keeps
residual recompile stalls in and is what a finite run delivers.

**Weak** — FSDP `shard=-1`, 32,768 tokens per DP rank, global batch grows:

| nodes | GPUs | tok/s/GPU | MFU | eff |
|---|---|---|---|---|
| 4 | 16 | 14,369 | 21.3% | 100% |
| 8 | 32 | 13,831 | 20.5% | 96.3% |
| 16 | 64 | 13,000 | 19.4% | 90.5% |
| 32 | 128 | 12,172 | 18.1% | 84.7% |
| 64 | 256 | 11,688 | 17.4% | 81.3% |
| 128 | 512 | 10,956 | 16.4% | **76.2%** |

**76.2% over a 32x range**, no cliff anywhere up to 512 GPUs.

**Strong** — HSDP (`replicate=nodes, shard=2`), global batch pinned at 2,097,152
tokens, accumulation 8/4/2/1:

| nodes | GPUs | accum | tok/s/GPU | MFU | eff |
|---|---|---|---|---|---|
| 4 | 16 | 8 | 14,201 | 21.2% | 100% |
| 8 | 32 | 4 | **15,302** | 22.8% | 107.8% |
| 16 | 64 | 2 | 14,052 | 20.9% | 99.0% |
| 32 | 128 | 1 | 13,026 | 19.4% | **91.7%** |

Strong beats weak at every shared node count and the gap widens: HSDP keeps both
sharding collectives on NVLink inside a node and sends only the gradient
all-reduce between nodes. It costs ~86 GiB peak against ~74.

The 32-GPU point being *above* the 16-GPU one is not an anomaly — at 16 GPUs
each rank runs 8 micro-batches per step against 4, so the left edge carries the
most accumulation overhead per token. Read that curve from its peak.

### `steptime_4n.png`, `steptime_8n.png`, `steptime_16n.png`, `steptime_32n.png`
Step-time distributions and stall decay. Step times are bimodal: a step either
runs at full rate or blocks on a `torch.compile` recompile somewhere in the
world. Each rank recompiles on its own shape schedule but every rank blocks at
the next collective, so the stall rate follows the *union* over ranks and grows
with world size.

**Warmup roughly doubles per doubling of the machine:**

| nodes | warmup ends | residual stalls | full-rate step |
|---|---|---|---|
| 4 | step 51 | 0.0% | 1.12 s |
| 8 | step 72 | 0.9% | 1.03 s |
| 16 | step 181 | 1.6% | 1.12 s |
| 32 | step 397 | 3.8% | 1.22 s |

**A benchmark at 32 nodes needs >= 500 steps to report anything meaningful.** The
same configuration reads 3,704 tok/s/GPU over 40 steps and 12,778 over 500.

### `precompile_32n_control.png`, `precompile_32n_enabled.png`
The same 32-node run with and without warming the compiler on synthetic shapes
before step 1.

| | warm at | elapsed to warm | lost vs full rate | steady tok/s/GPU |
|---|---|---|---|---|
| control | step 416 | 16.8 min | **7.8 min** | 12,221 |
| precompile | step 1 | 0.0 min | **0.0 min** | 12,222 |

Steady throughput is identical to four significant figures — it buys warmup and
nothing else. That is worth 7.8 min at 128 GPUs, 19.0 at 256 and 27.1 at 512,
and essentially nothing at one node.

### `packed_batch_shapes_plotqa.png`
Distribution of image counts and patch counts per packed row. This is the input
that drives the recompiles above: every new combination of sequence length,
image count and varlen metadata is a new graph.