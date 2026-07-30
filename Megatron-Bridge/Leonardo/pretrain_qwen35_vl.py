#!/usr/bin/env python3


#Developed from the Megatron-Bridge nvidia repo 


# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-VL MegatronMIMO pretraining runner (random init, no checkpoint).

This is the pretraining counterpart to ``finetune_qwen35_vl.py``. Where the SFT
runner loads a converted MegatronMIMO checkpoint and computes an
assistant-masked loss, this runner:

  - **Starts from random weights.** Both towers (language + vision encoder) are
    randomly initialized by the MegatronMIMO provider -- exactly as standard
    Megatron LLM pretraining works. No ``--pretrained-checkpoint`` is loaded;
    ``--load-checkpoint`` is still honored for resuming an interrupted run.
  - **Uses a full-sequence next-token objective.** Labels are the input tokens
    shifted by one and the loss mask covers every non-pad token (visual
    placeholder tokens and skipped special tokens are excluded). There is no
    prompt/assistant masking.

Two data modes are supported:

  - ``--data-mode hf``  (default): direct Hugging Face VLM data (e.g. ``cord_v2``
    preset, or a custom ``--dataset-path`` with a ``--schema-adapter``). The
    conversation is rendered to text and trained full-sequence. This exercises
    the pretraining *training path*; genuine pretraining should point
    ``--dataset-path`` at a caption / interleaved image-text corpus.
  - ``--data-mode mock``: synthetic image-text batches with no dataset download,
    for offline and CI smoke tests.

Everything downstream of label construction -- the MegatronMIMO provider, the
heterogeneous-parallelism forward step, MRoPE position ids, and the visual
reorganization -- is shared with the SFT runner and reused unchanged.

Example 2-GPU mock smoke:

  FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 \\
  uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \\
    examples/megatron_mimo/qwen35_vl/pretrain_qwen35_vl.py \\
      --data-mode mock \\
      --hf-model Qwen/Qwen3.5-0.8B \\
      --component language=tp=1,dp=1,rank_offset=0 \\
      --component images=tp=1,dp=1,rank_offset=1 \\
      --train-iters 2 --seq-length 256
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from transformers import AutoConfig


# The SFT runner lives next to this file; make the sibling import resolve whether
# this script is executed directly (dir already on sys.path[0]) or imported by a
# test via importlib.util.spec_from_file_location.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the loss-agnostic machinery from the SFT runner. These helpers build the
# provider, adapt an HF batch into the MIMO forward shape, parse the component
# layout, and select which fields a given rank needs -- none of which depends on
# the loss being SFT vs pretraining.
from finetune_qwen35_vl import (  # noqa: E402
    G_DEFAULT_COMPONENTS,
    G_EXAMPLE_ROOT,
    MIMOBatchSpec,
    Qwen35MIMOHFSpec,
    _adapt_qwen35_hf_batch,
    _batch_spec_for_rank,
    _build_dataset_config,
    _build_hf_spec,
    _build_mimo_provider,
    _build_parallelism_config,
    _expand_qwen_image_placeholders,
    _iter_image_parts,
    _parse_profile_ranks,
    _qwen_image_grid_for_part,
    _str2bool,
    _summarize_adapted_batch,
    _validate_mimo_batch_sizes,
)

from megatron.bridge.data.base import DatasetBuildContext  # noqa: E402
from megatron.bridge.data.builders import (  # noqa: E402
    DirectHFSFTDatasetBuilder,
    DirectHFSFTDatasetConfig,
    MockVLMSFTDatasetBuilder,
    MockVLMSFTDatasetConfig,
)
from megatron.bridge.data.conversation_processing import chat_template_kwargs_from_example  # noqa: E402
from megatron.bridge.data.datasets.utils import IGNORE_INDEX  # noqa: E402
from megatron.bridge.data.megatron_mimo.dp_utils import get_megatron_mimo_sampling_info  # noqa: E402
from megatron.bridge.data.samplers import build_pretraining_data_loader  # noqa: E402
from megatron.bridge.data.token_utils import extract_skipped_token_ids  # noqa: E402
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (  # noqa: E402
    MegatronMIMOParallelismConfig,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing  # noqa: E402
from megatron.bridge.training.config import (  # noqa: E402
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    ProfilingConfig,
    TrainingConfig,
)
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step  # noqa: E402
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo  # noqa: E402
from megatron.bridge.training.state import TrainState  # noqa: E402
from megatron.bridge.training.tokenizers.config import TokenizerConfig  # noqa: E402
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs  # noqa: E402


G_RANK_LOG_FILE = None


def _log(message: str) -> None:
    """Write a rank-prefixed message to stdout and the per-rank log file."""
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {message}\n"
    if G_RANK_LOG_FILE is not None:
        G_RANK_LOG_FILE.write(line)
        G_RANK_LOG_FILE.flush()
    print(line, end="", flush=True)


def _resolve_loss_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor | None, pad_token_id: int
) -> torch.Tensor:
    """Non-pad token mask, preferring the tokenizer attention mask when present."""
    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
        return attention_mask.to(dtype=torch.float32)
    return (input_ids != pad_token_id).to(dtype=torch.float32)


def _build_pretrain_labels_and_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    spec: Qwen35MIMOHFSpec,
    skipped_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full-sequence next-token labels and a non-pad loss mask.

    Unlike the SFT collate (which masks everything but assistant turns), the loss
    covers every real text token. Visual placeholder tokens and skipped special
    tokens are excluded because they are not prediction targets. Mirrors the
    labeling in ``data/megatron_mimo/dataset.py``.

    Args:
        input_ids: ``[batch, seq]`` token ids from the collate.
        attention_mask: Optional ``[batch, seq]`` padding mask.
        spec: Qwen3.5-VL constants (pad/image/video token ids).
        skipped_tokens: Special token ids to exclude from the loss.

    Returns:
        ``(labels, loss_mask)`` aligned for next-token prediction; ``labels`` is
        long with ``IGNORE_INDEX`` on masked positions, ``loss_mask`` is float32.
    """
    loss_mask = _resolve_loss_mask(input_ids, attention_mask, spec.pad_token_id)

    # Shift left so position t predicts token t+1; the last position has no target.
    labels = torch.cat(
        [input_ids[:, 1:], torch.full_like(input_ids[:, :1], IGNORE_INDEX)],
        dim=1,
    ).contiguous()
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)

    visual_targets = (labels == spec.image_token_id) | (labels == spec.video_token_id)
    loss_mask = loss_mask.masked_fill(visual_targets, 0.0)
    if skipped_tokens.numel() > 0:
        loss_mask = loss_mask.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), 0.0)

    labels = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    return labels, loss_mask.to(dtype=torch.float32)


def _build_qwen_pretrain_metadata_batch(
    items: list[Any],
    *,
    processor: Any,
    spec: Qwen35MIMOHFSpec,
    skipped_tokens: torch.Tensor,
    min_pixels: int,
    max_pixels: int,
) -> dict[str, Any]:
    """Metadata-only pretraining batch for text ranks that do not need pixels.

    Renders each conversation to text, expands the Qwen image placeholders from
    image-grid metadata (no pixel decode), tokenizes, and builds a full-sequence
    loss. Reuses the SFT runner's grid/placeholder helpers so the token layout
    matches the visual path exactly.
    """
    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    if image_processor is None:
        raise ValueError("Qwen metadata-only pretrain collate requires processor.image_processor.")

    image_token = getattr(processor, "image_token", "<|image_pad|>")
    merge_size = int(getattr(image_processor, "merge_size", spec.spatial_merge_size))
    texts: list[str] = []
    per_sample_grids: list[list[tuple[int, int, int]]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Qwen metadata-only pretrain collate expects dict conversation examples.")
        text = processor.apply_chat_template(
            item["conversation"],
            tokenize=False,
            **chat_template_kwargs_from_example(item),
        )
        grids = [
            _qwen_image_grid_for_part(part, image_processor, min_pixels=min_pixels, max_pixels=max_pixels)
            for part in _iter_image_parts(item)
        ]
        texts.append(_expand_qwen_image_placeholders(text, grids, image_token=image_token, merge_size=merge_size))
        per_sample_grids.append(grids)

    tokenized = tokenizer(texts, padding=True, return_tensors="pt", return_token_type_ids=False)
    input_ids = tokenized["input_ids"].contiguous()
    attention_mask = tokenized.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.contiguous()

    labels, loss_mask = _build_pretrain_labels_and_mask(input_ids, attention_mask, spec, skipped_tokens)

    flat_grids = [grid for grids in per_sample_grids for grid in grids]
    image_grid_thw = torch.tensor(flat_grids, dtype=torch.long) if flat_grids else None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
        "visual_inputs": GenericVisualInputs(image_grid_thw=image_grid_thw),
    }


class _Qwen35HFPretrainCollateAdapter:
    """Wrap the dataset's collate, then rebuild labels/loss_mask full-sequence.

    Runs the standard HF/mock VLM collate (producing ``input_ids`` and
    ``visual_inputs``), replaces the assistant-masked labels with a full-sequence
    next-token objective, and adapts the batch into the MIMO forward shape.
    """

    def __init__(
        self,
        base_collate: Callable[[list[Any]], dict[str, Any]],
        processor: Any,
        spec: Qwen35MIMOHFSpec,
        seq_length: int,
        pad_to_seq_length: bool,
        batch_spec: MIMOBatchSpec,
    ) -> None:
        self.base_collate = base_collate
        self.spec = spec
        self.seq_length = seq_length
        self.pad_to_seq_length = pad_to_seq_length
        self.batch_spec = batch_spec
        self.skipped_tokens = extract_skipped_token_ids(processor)

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        batch = self.base_collate(items)
        input_ids = batch.get("tokens") if batch.get("tokens") is not None else batch["input_ids"]
        labels, loss_mask = _build_pretrain_labels_and_mask(
            input_ids, batch.get("attention_mask"), self.spec, self.skipped_tokens
        )
        batch["labels"] = labels
        batch["loss_mask"] = loss_mask
        return _adapt_qwen35_hf_batch(
            batch,
            self.spec,
            seq_length=self.seq_length,
            pad_to_seq_length=self.pad_to_seq_length,
            batch_spec=self.batch_spec,
        )


class _Qwen35HFPretrainMetadataCollateAdapter:
    """Qwen metadata-only pretraining collate for ranks that skip image tensors."""

    def __init__(
        self,
        processor: Any,
        spec: Qwen35MIMOHFSpec,
        seq_length: int,
        pad_to_seq_length: bool,
        batch_spec: MIMOBatchSpec,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        from megatron.bridge.models.qwen_vl.data.collate_fn import QWEN_VL_MAX_PIXELS, QWEN_VL_MIN_PIXELS

        self.processor = processor
        self.spec = spec
        self.seq_length = seq_length
        self.pad_to_seq_length = pad_to_seq_length
        self.batch_spec = batch_spec
        self.min_pixels = QWEN_VL_MIN_PIXELS if min_pixels is None else min_pixels
        self.max_pixels = QWEN_VL_MAX_PIXELS if max_pixels is None else max_pixels
        self.skipped_tokens = extract_skipped_token_ids(processor)

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        batch = _build_qwen_pretrain_metadata_batch(
            items,
            processor=self.processor,
            spec=self.spec,
            skipped_tokens=self.skipped_tokens,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return _adapt_qwen35_hf_batch(
            batch,
            self.spec,
            seq_length=self.seq_length,
            pad_to_seq_length=self.pad_to_seq_length,
            batch_spec=self.batch_spec,
        )


def _wrap_iter_logging(loader_iter: Iterator[dict[str, Any]], spec: Qwen35MIMOHFSpec) -> Iterator[dict[str, Any]]:
    for batch_idx, adapted in enumerate(loader_iter):
        _log(f"pretrain batch {batch_idx}: {_summarize_adapted_batch(adapted, spec)}")
        yield adapted


def _build_train_dataset(cfg: Any, context: DatasetBuildContext) -> Any:
    """Build the train split for either the HF or the mock data mode."""
    if isinstance(cfg.dataset, MockVLMSFTDatasetConfig):
        train_ds, _, _ = MockVLMSFTDatasetBuilder(cfg.dataset).build(context)
    elif isinstance(cfg.dataset, DirectHFSFTDatasetConfig):
        train_ds, _, _ = DirectHFSFTDatasetBuilder(cfg.dataset).build(context)
    else:
        raise TypeError(
            "MegatronMIMO Qwen3.5-VL pretraining requires DirectHFSFTDatasetConfig or MockVLMSFTDatasetConfig."
        )
    if train_ds is None:
        raise ValueError("Dataset builder did not build a train dataset.")
    return train_ds


def _make_build_data_iterators(spec: Qwen35MIMOHFSpec, args: argparse.Namespace):
    def _build_data_iterators(cfg, _megatron_mimo_infra, *, train_state=None):
        if train_state is None:
            train_state = TrainState()

        if cfg.model._grids is None:
            raise ValueError("MegatronMIMOProvider._grids is None. Model must be built before data iterators.")

        sampler_dp_rank, sampler_dp_size, needs_data = get_megatron_mimo_sampling_info(
            cfg.model.megatron_mimo_parallelism_config,
            cfg.model._grids,
        )
        if not needs_data:
            return None, None

        train_samples = max(cfg.train.train_iters * cfg.train.global_batch_size, 10)
        context = DatasetBuildContext(
            train_samples=train_samples,
            valid_samples=0,
            test_samples=0,
            tokenizer=None,
        )
        train_ds = _build_train_dataset(cfg, context)
        base_collate = getattr(train_ds, "collate_fn", None)
        if base_collate is None:
            raise ValueError("Train dataset does not expose collate_fn.")
        processor = getattr(train_ds, "_processor", None)
        if processor is None:
            raise ValueError("Train dataset does not expose an HF processor.")

        batch_spec = _batch_spec_for_rank(cfg)
        use_metadata_collate = not batch_spec.modality_inputs
        _log(
            f"mimo_batch_spec spec={batch_spec.describe()} collate={'metadata' if use_metadata_collate else 'visual'}"
        )

        if use_metadata_collate:
            collate_fn: Callable[[list[Any]], dict[str, Any]] = _Qwen35HFPretrainMetadataCollateAdapter(
                processor=processor,
                spec=spec,
                seq_length=args.seq_length,
                pad_to_seq_length=args.pad_to_seq_length,
                batch_spec=batch_spec,
            )
        else:
            collate_fn = _Qwen35HFPretrainCollateAdapter(
                base_collate=base_collate,
                processor=processor,
                spec=spec,
                seq_length=args.seq_length,
                pad_to_seq_length=args.pad_to_seq_length,
                batch_spec=batch_spec,
            )

        train_loader = build_pretraining_data_loader(
            dataset=train_ds,
            consumed_samples=train_state.consumed_train_samples,
            dataloader_type=cfg.dataset.dataloader_type,
            micro_batch_size=cfg.train.micro_batch_size,
            num_workers=cfg.dataset.num_workers,
            data_sharding=getattr(cfg.dataset, "data_sharding", True),
            collate_fn=collate_fn,
            pin_memory=getattr(cfg.dataset, "pin_memory", True),
            persistent_workers=getattr(cfg.dataset, "persistent_workers", False),
            data_parallel_rank=sampler_dp_rank,
            data_parallel_size=sampler_dp_size,
            drop_last=getattr(cfg.dataset, "drop_last", True),
        )

        loader_iter: Iterator[dict[str, Any]] = iter(train_loader)
        if args.log_batches:
            loader_iter = _wrap_iter_logging(loader_iter, spec)
        return loader_iter, None

    return _build_data_iterators


def _build_dataset(args: argparse.Namespace) -> DirectHFSFTDatasetConfig | MockVLMSFTDatasetConfig:
    if args.data_mode == "mock":
        dataset_config = MockVLMSFTDatasetConfig(
            seq_length=args.seq_length,
            hf_processor_path=args.processor_path or args.hf_model,
            num_images=args.mock_num_images,
            image_size=(args.mock_image_size, args.mock_image_size),
            dataloader_type=args.dataloader_type,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            trust_remote_code=args.trust_remote_code,
        )
        dataset_config.drop_last = True
        return dataset_config
    return _build_dataset_config(args)


def _build_checkpoint_config(args: argparse.Namespace) -> CheckpointConfig:
    checkpoint_cfg = CheckpointConfig()
    checkpoint_cfg.save = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        checkpoint_cfg.save_interval = args.checkpoint_interval
    if args.load_checkpoint is not None:
        checkpoint_cfg.load = args.load_checkpoint
    checkpoint_cfg.ckpt_format = "torch_dist"
    checkpoint_cfg.fully_parallel_save = True
    checkpoint_cfg.dist_ckpt_optim_fully_reshardable = True
    checkpoint_cfg.save_rng = False
    return checkpoint_cfg


def _build_config(
    *,
    model_provider,
    dataset_config,
    args: argparse.Namespace,
) -> ConfigContainer:
    optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        clip_grad=args.clip_grad,
        start_weight_decay=args.start_weight_decay,
        end_weight_decay=args.end_weight_decay,
    )
    optimizer_cfg.bf16 = not args.fp32
    optimizer_cfg.fp16 = False
    optimizer_cfg.use_precision_aware_optimizer = False
    optimizer_cfg.main_grads_dtype = torch.float32
    optimizer_cfg.main_params_dtype = torch.float32
    optimizer_cfg.exp_avg_dtype = torch.float32
    optimizer_cfg.exp_avg_sq_dtype = torch.float32

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = args.log_interval
    logger_cfg.log_timers_to_tensorboard = True
    logger_cfg.tensorboard_dir = args.tensorboard_dir
    logger_cfg.wandb_project = args.wandb_project
    logger_cfg.wandb_exp_name = args.wandb_exp_name
    logger_cfg.wandb_entity = args.wandb_entity
    logger_cfg.wandb_save_dir = args.wandb_save_dir

    profiling_cfg = ProfilingConfig(
        use_nsys_profiler=args.profile == "nsys",
        use_pytorch_profiler=args.profile == "pytorch",
        profile_step_start=args.profile_step_start,
        profile_step_end=args.profile_step_end,
        profile_ranks=_parse_profile_ranks(args.profile_ranks),
        record_shapes=args.profile_record_shapes,
        pytorch_profiler_collect_shapes=args.profile_record_shapes,
        nvtx_ranges=args.profile_nvtx_ranges,
    )

    cfg = ConfigContainer(
        train=TrainingConfig(
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            train_iters=args.train_iters,
            eval_interval=None,
            eval_iters=None,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        model=model_provider,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        dataset=dataset_config,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=_build_checkpoint_config(args),
        profiling=profiling_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        ),
    )
    cfg.data_parallel_size = 1
    cfg.rng.seed = args.seed
    cfg.mixed_precision = "bf16_mixed" if not args.fp32 else None
    return cfg


def _resolve_default_paths(args: argparse.Namespace) -> None:
    model_tag = Path(args.hf_model.rstrip("/")).name
    # Pretraining starts from random init; only a checkpoint attr is needed so the
    # shared config helpers can treat "no pretrained weights" uniformly.
    args.pretrained_checkpoint = None
    if args.data_mode == "hf" and args.dataset_name is None and args.dataset_path is None:
        if args.dataset_subset is not None or args.schema_adapter is not None:
            raise ValueError("--dataset-subset and --schema-adapter require --dataset-path.")
        args.dataset_name = "cord_v2"
    if args.data_mode == "hf" and args.dataset_name is not None and args.dataset_path is not None:
        raise ValueError("Set either --dataset-name or --dataset-path, not both.")
    if args.checkpoint_dir is None:
        dataset_tag = args.dataset_name or args.schema_adapter or args.data_mode
        run_name = args.run_name or f"{model_tag}_{dataset_tag}_mimo_pretrain"
        args.checkpoint_dir = str(Path(args.experiment_root) / "results" / "mimo_pretrain" / run_name)
    if args.log_dir is None:
        args.log_dir = str(Path(args.experiment_root) / "logs" / "mimo_pretrain")
    if args.tensorboard_dir is None:
        args.tensorboard_dir = str(Path(args.checkpoint_dir) / "tb_logs")
    if args.wandb_save_dir is None:
        args.wandb_save_dir = str(Path(args.checkpoint_dir) / "wandb")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MegatronMIMO Qwen3.5-VL random-init multimodal pretraining")
    parser.add_argument("--hf-model", type=str, default="Qwen/Qwen3.5-0.8B", help="HF model id or local config path")
    parser.add_argument("--processor-path", type=str, default=None, help="HF processor path; defaults to --hf-model")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help="Component layout: name=tp=N[,pp=N,cp=N,dp=N,rank_offset=N]",
    )
    parser.add_argument("--experiment-root", type=str, default=G_EXAMPLE_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--data-mode",
        choices=("hf", "mock"),
        default="hf",
        help="hf: direct Hugging Face VLM data; mock: synthetic image-text batches (offline/CI smoke).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Built-in HF dataset preset (data-mode hf). Defaults to cord_v2 when no custom path is set.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Custom HF dataset path (data-mode hf); mutually exclusive with --dataset-name.",
    )
    parser.add_argument("--dataset-subset", type=str, default=None)
    parser.add_argument("--schema-adapter", type=str, default=None, help="Adapter for a custom non-native source.")
    parser.add_argument(
        "--do-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable a derived validation split; presets auto-enable it only when supported.",
    )
    parser.add_argument("--mock-num-images", type=int, default=1, help="Images per synthetic sample (data-mode mock).")
    parser.add_argument("--mock-image-size", type=int, default=256, help="Square image size (data-mode mock).")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataloader-type", choices=("single", "cyclic"), default="cyclic")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of bf16")
    # From-scratch pretraining trains every tower by default.
    parser.add_argument("--freeze-vision", type=_str2bool, default=False)
    parser.add_argument("--freeze-llm", type=_str2bool, default=False)
    parser.add_argument("--freeze-projector", type=_str2bool, default=False)
    # Pretraining-scale optimizer defaults (vs. the SFT runner's 5e-6).
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-warmup-iters", type=int, default=2000)
    parser.add_argument("--lr-decay-iters", type=int, default=300000)
    parser.add_argument("--start-weight-decay", type=float, default=0.1)
    parser.add_argument("--end-weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-exp-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint directory for full resume")
    parser.add_argument(
        "--pad-to-seq-length",
        type=_str2bool,
        default=True,
        help="Pad/truncate batches to --seq-length before the MIMO forward.",
    )
    parser.add_argument("--profile", choices=("none", "nsys", "pytorch"), default="none")
    parser.add_argument("--profile-step-start", type=int, default=1)
    parser.add_argument("--profile-step-end", type=int, default=2)
    parser.add_argument(
        "--profile-ranks",
        type=str,
        default="0",
        help="Comma-separated global ranks to profile, or 'all' for every rank.",
    )
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-nvtx-ranges", action="store_true")
    parser.add_argument("--log-batches", action="store_true", help="Log per-batch image/token summary.")
    args = parser.parse_args()
    if args.data_mode == "mock" and (
        args.dataset_name is not None or args.dataset_path is not None or args.schema_adapter is not None
    ):
        raise ValueError("--data-mode mock does not use --dataset-name/--dataset-path/--schema-adapter.")
    _resolve_default_paths(args)
    return args


def main() -> None:
    """Entry point for Qwen3.5-VL MegatronMIMO random-init pretraining."""
    global G_RANK_LOG_FILE

    args = _parse_args()
    components = args.component or G_DEFAULT_COMPONENTS
    if args.wandb_project is None:
        os.environ.setdefault("WANDB_MODE", "disabled")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    G_RANK_LOG_FILE = open(Path(args.log_dir) / f"rank_{rank}.log", "w")
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(args.log_dir) / f"rank_{rank}_full.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )

    succeeded = False
    try:
        _log(f"distributed initialized (world_size={dist.get_world_size()})")
        _log(f"loading HF config from {args.hf_model}")
        hf_config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        hf_spec = _build_hf_spec(hf_config)
        _log(
            f"qwen constants: image_token_id={hf_spec.image_token_id}, "
            f"vision_start_token_id={hf_spec.vision_start_token_id}, "
            f"spatial_merge_size={hf_spec.spatial_merge_size}"
        )

        parallelism_config: MegatronMIMOParallelismConfig = _build_parallelism_config(
            components, dist.get_world_size()
        )
        _log(f"component layout: {components}")
        for summary in _validate_mimo_batch_sizes(parallelism_config, args):
            _log(f"batch contract: global_mbs={args.micro_batch_size}, {summary}")

        _log("building Qwen3.5-VL MegatronMIMO provider (random init, no checkpoint)")
        model_provider = _build_mimo_provider(hf_config, parallelism_config, args)

        _log(f"building pretraining data: mode={args.data_mode}")
        dataset_config = _build_dataset(args)

        _log(f"checkpoint dir: {args.checkpoint_dir}")
        if args.load_checkpoint is not None:
            _log(f"resuming from checkpoint: {args.load_checkpoint}")
        else:
            _log("no checkpoint to load: training from randomly initialized weights")
        _log("building training config")
        cfg = _build_config(model_provider=model_provider, dataset_config=dataset_config, args=args)

        _log("launching pretrain_megatron_mimo")
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_make_build_data_iterators(hf_spec, args),
        )
        _log("PASSED")
        succeeded = True
    finally:
        if succeeded:
            dist.destroy_process_group()
        if G_RANK_LOG_FILE is not None:
            G_RANK_LOG_FILE.close()
            G_RANK_LOG_FILE = None


if __name__ == "__main__":
    main()
