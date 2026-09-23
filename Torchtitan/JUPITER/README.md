# TorchTitan on JUPITER

The second codebase benchmarked on JUPITER, against `VLM-TRAINING/JUPITER` on
the same machine, the same model (Qwen3.5-9B) and the same dataset.

JUPITER (JSC): GH200 nodes, **4 GPUs per node**, `booster` partition, aarch64.

## Pinned revision

[pytorch/torchtitan](https://github.com/pytorch/torchtitan) at **`b21f7d43e`**.

```bash
TT_ROOT=/path/to/torchtitan
git clone https://github.com/pytorch/torchtitan "$TT_ROOT/repo"
cd "$TT_ROOT/repo" && git checkout b21f7d43e
```

**The checkout is never edited.** A reference implementation you have patched is
not a reference. Everything in this directory lives outside it: the benchmark
configs are reached with `--module ttbench`, and the one unavoidable upstream
fix is applied at runtime (see *Patches*).

## Environment

torch comes from the same conda env `VLM-TRAINING` uses, so both codebases are
measured on one torch build. A venv on top supplies only torchtitan's own
dependency set:

```bash
source /e/project1/open-sci-mm/ockier1/envs/miniforge3/etc/profile.d/conda.sh
conda activate /e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main
python -m venv --system-site-packages "$TT_ROOT/venv"
source "$TT_ROOT/venv/bin/activate"
pip install -e "$TT_ROOT/repo"      # plus spmd_types, attn_gym
```

Activation order matters and is what keeps `torch_main` unmodified.

> **Never launch with the `torchrun` console script.** It lives in
> `torch_main/bin` and its shebang hard-pins the interpreter to
> `torch_main/bin/python`, which skips the venv's site-packages no matter what
> `activate` did to `PATH`. The job dies with `ModuleNotFoundError: No module
> named 'spmd_types'` *after* a pre-flight import in the same shell has just
> succeeded. Go through the venv interpreter and run the module:
> `$TT_ROOT/venv/bin/python -m torch.distributed.run ...`. `tt_bench.sbatch`
> already does this.

## Install the benchmark assets

```bash
mkdir -p "$TT_ROOT/bench"
cp config_registry.py chat_template_sft.jinja "$TT_ROOT/bench/"
cp -r patches "$TT_ROOT/bench/"
cp tt_bench.sbatch submit_tt_sweep.sh <your vlm-training>/scripts/scaling/
```

`tt_bench.sbatch` puts `$TT_ROOT/bench` on `PYTHONPATH`, which is how
`--module ttbench` resolves.

Tokenizer assets: symlink an HF Qwen3.5-9B snapshot to
`repo/assets/hf/Qwen3.5-9B`. **No weights needed**

## Run

```bash
# one point
TT_MODULE=ttbench TT_CONFIG=qwen35_9b_plotqa \
  sbatch --nodes=4 tt_bench.sbatch --training.disable-cuda-graphs

# a sweep
./submit_tt_sweep.sh 4 8 16 32
```

Node count comes from `sbatch --nodes`, never from a directive. Everything after
the script name is forwarded to `torchtitan.train` verbatim.

## Configs — `config_registry.py`

| function | seq_len | purpose |
|---|---|---|
| `qwen35_9b_local` | 10240 | single node, their `cc12m-test` data |
| `qwen35_9b_plotqa` | 10240 | our `plotqa_cot` shards, packed |
| `qwen35_9b_plotqa_16384` | 16384 | the strong/weak scaling workload |
| `qwen35_9b_plotqa_{16896,17408,17920,18176}` | as named | seq_len grown into the memory FSDP frees at 4/8/16/32 nodes |
| `qwen35_2b_plotqa_local`, `qwen35_9b_plotqa_local*` | 8192-16384 | local Blackwell box, via `_blackwell_shims()` |

**Only two settings cannot be CLI overrides**, which is the whole reason this
file exists:

- **`seq_len`** — `config_fn()` is called with no arguments, and
  `model_registry(flavor, seq_len=...)` feeds it into `get_config()` itself.
  Overriding `--training.max-context-length` builds a 262144 model and runs
  10240 rows through it, which measures something else.
- **`dataset`** — `MM_DATASETS` entries carry `Suppress`-ed callables, so tyro
  will not construct one from the command line.

Everything else is a flag: `--parallelism.tensor-parallel-degree`,
`--parallelism.fsdp-reshard-after-forward never`, `--activation-checkpoint none`,
`--compile.enable`, `--training.num-tokens-per-microbatch-per-dp-rank`.

> for the configs, write me: `tockier@cvc.uab.cat`

### Deltas from the shipped `qwen35_9b`

It is a pretraining recipe, not a benchmark config. Left alone it measures
something else:

| | benchmark | `qwen35_9b` default |
|---|---|---|
| sequence length | 10240 / 16384 | **262144** |
| tokens/microbatch/rank | 1 row | **4 x** max_context_length |
| TP degree | 4 | 2 |
| `reshard_after_forward` | `never` | default |
| activation checkpointing | **none** | **FullAC** |
| packing | on, ~99% occupancy | **off** |

The last two decide the result on their own. `FullAC` is worth a large fraction
of a step, and unpacked rows are one document padded to `max_context_length` —
comparing a 99%-full row against a padded one compares padding, not
implementations.

### Data

`qwen35_9b` ships `MM_DATASETS["cc12m"]`, an `HuggingFaceStreamingSource` pull
from the hub. Compute nodes have no internet, so it is dead on arrival.
`config_registry.py` instead points a `SingleDatasetConfig` at a local directory
of WebDataset tars — the `{key}.png` + `{key}.json` shards that
`utils/prepare_nemotron_energon.py` already writes — and wraps it in
`MMSamplePackingConfig` so occupancy matches.

## Patches

`patches/attn_gym_gdn_fake_strides.py` — **required on sm90.** The `attn_gym`
GatedDeltaNet custom op ships a wrong meta kernel; `torch.compile` trusts it and
the run dies tens of steps in, at a scale-dependent moment, with an assertion
about strides rather than anything resembling its cause. Applied at runtime, not
committed into the checkout.

## Mandatory flag

```
--training.disable-cuda-graphs
```

Packed multimodal rows carry a different number of images each step, so the
pixel tensor changes shape and capture fails on step 2:

```
ValueError: CUDA graph tensor inputs must keep the same shape, dtype, and
device, but input 4 changed from (1344, 1536) to (1040, 1536)
```

## Results

All figures in `plots/`. Qwen3.5-9B, plotqa_cot, `tp=4`, 500 steps, figures taken
over the last 100.

### `tt_weak_scaling.png`
FSDP `shard=-1`, **32,768 tokens per DP rank** held fixed, so the global batch
grows with the machine.

| nodes | GPUs | tok/s/GPU | MFU | eff | peak |
|---|---|---|---|---|---|
| 4 | 16 | 4,864 | 26.7% | 100% | 85.3 GiB |
| 8 | 32 | 4,553 | 25.0% | 93.6% | 82.9 GiB |
| 16 | 64 | 4,351 | 23.9% | 89.5% | 81.1 GiB |
| 32 | 128 | 4,269 | 23.4% | **87.8%** | 79.8 GiB |

### `tt_strong_scaling_ddp.png`
`replicate=N, shard=1` — DDP in all but name — with the **global batch pinned at
524,288 tokens**, so accumulation absorbs the scale (8/4/2/1 micro-batches).

| nodes | GPUs | accum | tok/s/GPU | eff |
|---|---|---|---|---|
| 4 | 16 | 8 | 4,596 | 100% |
| 32 | 128 | 1 | 3,031 | **66.0%** |

### `tt_weak_scaling_growing_seqlen.png`
As the weak sweep, but `seq_len` grows into the memory FSDP frees as the
optimizer shards thin out: 16896 / 17408 / 17920 / 18176 at 4 / 8 / 16 / 32
nodes. 4,927 -> 4,342 tok/s/GPU, **88.1%** — the same 32-node efficiency as the
constant-`seq_len` sweep. FSDP frees only 5.5 GiB from 4 to 32 nodes, which buys
+7.6% tokens/rank and cannot offset the collective cost.

### `steptime_tt_weak_16n.png`, `steptime_tt_weak_32n.png`, `steptime_tt_grow_32n.png`
Per-step time distributions. Step times are bimodal, not noisy: a step either
runs at full rate or waits on a recompile. The stall count grows with scale
(2 -> 10 per 100 steps from 4 to 32 nodes), which is why a short run at high node
count reports warmup rather than throughput.

### `qwen3_5_9b_vs_vlm_training.png`
The head-to-head against `VLM-TRAINING/JUPITER`, at matched tokens per step
(32,768 per DP rank) on the same machine, model and dataset. Solid lines are
full-rate steps, dashed the post-warmup mean.

| | VLM-TRAINING | TorchTitan |
|---|---|---|
| 16 GPUs, full-rate | 4,894 | 4,864 |
| 128 GPUs, full-rate | 4,490 | 4,269 |
| eff 16 -> 128 | 91.7% | 87.8% |

Near-parity per step. The MFU annotations are each framework's own and are **not
comparable across the two** — see below.

## Reading the results

**Warmup dominates any short run, and grows with world size.** A 40-step
benchmark at 32 nodes measures warmup and nothing else — the same configuration
read 3,704 tok/s/GPU over 40 steps and 8,373 over 500. Use >= 500 steps at 32
nodes and report the mean over steps after warmup, with the residual stall rate
beside it.

**Do not compare MFU or TFLOP/s across the two codebases.** TorchTitan bills a
constant ~54.29 GFLOP/token against the full row; `VLM-TRAINING` counts
per-document causal pairs plus the actual ViT patch count. They are different
measurements. **tok/s/GPU at matched tokens per step** is the quantity that
survives the comparison.