"""Fit a custom FAST action tokenizer from an openpi training config."""

# Monkey-patch to fix 'List' feature type error in old datasets.
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass
# End of monkey-patch.

import inspect
import json
import math
import multiprocessing
import pathlib
import shutil
import typing

import jax
import numpy as np
from scipy.fft import dct
import torch
import tqdm
from transformers import AutoProcessor
import tyro

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


_FAST_TOKENIZER_SOURCE = "physical-intelligence/fast"
_DEFAULT_EVAL_CHUNKS = 256
_MIN_BPE_MERGE_ROOM = 100


class KeepActions(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"], dtype=np.float32)}


def _create_action_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    if config.model.model_type != _model.ModelType.PI0_FAST:
        raise ValueError(f"FAST tokenizer fitting requires a pi0_FAST config, got {config.model.model_type}.")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id.")

    norm_stats = {}
    if data_config.repo_id != "fake":
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run "
                f"`python scripts/compute_norm_stats_fast.py --config-name={config.name}` first."
            )
        norm_stats = data_config.norm_stats

    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            KeepActions(),
        ],
    )
    return data_config, dataset


def _collate_fn(items):
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _create_torch_loader(dataset: _data_loader.Dataset, *, batch_size: int, num_workers: int, shuffle: bool):
    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        drop_last=False,
        generator=generator,
    )


def _collect_action_chunks(
    dataset: _data_loader.Dataset,
    *,
    max_chunks: int | None,
    batch_size: int,
    num_workers: int,
) -> list[np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot fit a FAST tokenizer.")

    target_chunks = len(dataset) if max_chunks is None else min(max_chunks, len(dataset))
    local_batch_size = min(batch_size, target_chunks)
    shuffle = max_chunks is not None and max_chunks < len(dataset)
    loader = _create_torch_loader(dataset, batch_size=local_batch_size, num_workers=num_workers, shuffle=shuffle)

    action_chunks: list[np.ndarray] = []
    total_batches = math.ceil(target_chunks / local_batch_size)
    for batch in tqdm.tqdm(loader, total=total_batches, desc="Collecting normalized action chunks"):
        actions = np.asarray(batch["actions"], dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [batch, horizon, dim], got {actions.shape}.")

        remaining = target_chunks - len(action_chunks)
        action_chunks.extend(np.asarray(chunk, dtype=np.float32) for chunk in actions[:remaining])
        if len(action_chunks) >= target_chunks:
            break

    if not action_chunks:
        raise ValueError("No action chunks were collected; cannot fit a FAST tokenizer.")

    action_array = np.stack(action_chunks, axis=0)
    if not np.all(np.isfinite(action_array)):
        raise ValueError("Collected action chunks contain NaN or Inf values.")

    return action_chunks


def _token_lengths(processor, action_chunks: list[np.ndarray], *, batch_size: int) -> np.ndarray:
    lengths = []
    for start in range(0, len(action_chunks), batch_size):
        batch = np.stack(action_chunks[start:start + batch_size], axis=0)
        lengths.extend(len(tokens) for tokens in processor(batch))
    return np.asarray(lengths, dtype=np.int32)


def _print_length_summary(name: str, lengths: np.ndarray, *, max_token_len: int) -> None:
    over_model_len = int(np.sum(lengths > max_token_len))
    print(
        f"{name}: mean_action_tokens={float(np.mean(lengths)):.2f}, "
        f"p95_action_tokens={float(np.percentile(lengths, 95)):.2f}, "
        f"max_action_tokens={int(np.max(lengths))}, "
        f"chunks_with_action_tokens_gt_model_max_len={over_model_len}/{len(lengths)}"
    )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _min_vocab_size(action_chunks: list[np.ndarray], *, scale: float) -> int:
    dct_tokens = [dct(action, axis=0, norm="ortho").flatten() for action in action_chunks]
    quantized = np.around(np.concatenate(dct_tokens) * scale)
    return int(quantized.max() - quantized.min())


def _resolve_vocab_size(action_chunks: list[np.ndarray], *, requested_vocab_size: int, scale: float) -> int:
    min_vocab_size = _min_vocab_size(action_chunks, scale=scale)
    recommended_vocab_size = _next_power_of_two(min_vocab_size + _MIN_BPE_MERGE_ROOM)
    if requested_vocab_size >= min_vocab_size + _MIN_BPE_MERGE_ROOM:
        return requested_vocab_size

    resolved_vocab_size = max(requested_vocab_size, recommended_vocab_size)
    print(
        f"Requested vocab_size={requested_vocab_size} is too small for this action range "
        f"(minimum={min_vocab_size}, recommended>={min_vocab_size + _MIN_BPE_MERGE_ROOM}); "
        f"using vocab_size={resolved_vocab_size}."
    )
    return resolved_vocab_size


def _round_trip_check(processor, action_chunks: list[np.ndarray], *, action_horizon: int, action_dim: int) -> None:
    eval_chunks = np.stack(action_chunks[:min(_DEFAULT_EVAL_CHUNKS, len(action_chunks))], axis=0)
    tokens = processor(eval_chunks)
    decoded = processor.decode(tokens, time_horizon=action_horizon, action_dim=action_dim)

    if decoded.shape != eval_chunks.shape:
        raise ValueError(f"Decoded actions have shape {decoded.shape}, expected {eval_chunks.shape}.")
    if not np.all(np.isfinite(decoded)):
        raise ValueError("Decoded actions contain NaN or Inf values.")

    mse = float(np.mean((decoded - eval_chunks)**2))
    max_abs = float(np.max(np.abs(decoded - eval_chunks)))
    print(f"Custom tokenizer round-trip: mse={mse:.6f}, max_abs_error={max_abs:.6f}, chunks={len(eval_chunks)}")


def _copy_processor_module(processor, output_dir: pathlib.Path) -> None:
    target = output_dir / _tokenizer._FAST_PROCESSING_MODULE
    if target.is_file():
        return

    candidates: list[pathlib.Path] = []
    if source_dir := _tokenizer._resolve_snapshot_dir(_FAST_TOKENIZER_SOURCE):
        candidates.append(source_dir / _tokenizer._FAST_PROCESSING_MODULE)

    if source_file := inspect.getsourcefile(processor.__class__):
        candidates.append(pathlib.Path(source_file))

    for candidate in candidates:
        if candidate.is_file():
            shutil.copy2(candidate, target)
            return

    raise FileNotFoundError(
        f"Could not find {_tokenizer._FAST_PROCESSING_MODULE}; "
        f"expected one of: {', '.join(str(c) for c in candidates)}"
    )


def _write_processor_config(processor, output_dir: pathlib.Path, *, action_horizon: int, action_dim: int) -> None:
    config_path = output_dir / _tokenizer._FAST_PROCESSOR_CONFIG
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            processor_config = json.load(f)
    else:
        processor_config = {}

    processor_config.update({
        "action_dim": action_dim,
        "auto_map": {
            "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
        },
        "min_token": processor.min_token,
        "processor_class": "UniversalActionProcessor",
        "scale": processor.scale,
        "time_horizon": action_horizon,
        "vocab_size": processor.vocab_size,
    })
    config_path.write_text(json.dumps(processor_config, indent=2) + "\n", encoding="utf-8")


def _validate_saved_tokenizer(output_dir: pathlib.Path) -> None:
    required_files = (
        _tokenizer._FAST_PROCESSING_MODULE,
        _tokenizer._FAST_PROCESSOR_CONFIG,
        "tokenizer.json",
    )
    missing = [name for name in required_files if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Saved FAST tokenizer is missing required files: {missing}")

    AutoProcessor.from_pretrained(str(output_dir), trust_remote_code=True, local_files_only=True)


def _load_base_processor():
    if source_dir := _tokenizer._resolve_snapshot_dir(_FAST_TOKENIZER_SOURCE):
        return AutoProcessor.from_pretrained(str(source_dir), trust_remote_code=True, local_files_only=True)
    return AutoProcessor.from_pretrained(_FAST_TOKENIZER_SOURCE, trust_remote_code=True)


def main(
    config_name: str,
    output_dir: pathlib.Path | None = None,
    max_chunks: int | None = None,
    batch_size: int = 256,
    num_workers: int = 8,
    vocab_size: int = 2048,
    scale: float = 10.0,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive when set, got {max_chunks}.")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

    config = _config.get_config(config_name)
    output_dir = output_dir or (config.assets_dirs / "fast_tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config, dataset = _create_action_dataset(config)
    action_chunks = _collect_action_chunks(
        dataset,
        max_chunks=max_chunks,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    action_array = np.stack(action_chunks, axis=0)
    print(
        f"Fitting FAST tokenizer from {len(action_chunks)} normalized action chunks "
        f"with shape {action_array.shape[1:]}."
    )
    print(f"Dataset repo_id: {data_config.repo_id}")
    print(f"Output dir: {output_dir}")

    base_processor = _load_base_processor()
    vocab_size = _resolve_vocab_size(action_chunks, requested_vocab_size=vocab_size, scale=scale)
    custom_processor = base_processor.fit(
        action_chunks,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )

    universal_lengths = _token_lengths(base_processor, action_chunks, batch_size=batch_size)
    custom_lengths = _token_lengths(custom_processor, action_chunks, batch_size=batch_size)
    _print_length_summary("Universal FAST", universal_lengths, max_token_len=config.model.max_token_len)
    _print_length_summary("Custom FAST", custom_lengths, max_token_len=config.model.max_token_len)
    if np.percentile(custom_lengths, 95) > np.percentile(universal_lengths, 95):
        print("Warning: custom FAST p95 action token count is higher than universal FAST; A/B before using it.")

    _round_trip_check(
        custom_processor,
        action_chunks,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )

    custom_processor.save_pretrained(output_dir)
    _copy_processor_module(custom_processor, output_dir)
    _write_processor_config(
        custom_processor,
        output_dir,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )
    _validate_saved_tokenizer(output_dir)

    print("Done. Use this tokenizer by setting:")
    print(f"export OPENPI_FAST_TOKENIZER_PATH={output_dir.resolve()}")


if __name__ == "__main__":
    tyro.cli(main)
