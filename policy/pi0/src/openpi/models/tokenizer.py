import json
import logging
import os
import pathlib
import types

from huggingface_hub import try_to_load_from_cache
from huggingface_hub.errors import HFValidationError
import numpy as np
import sentencepiece
from transformers import AutoProcessor, PreTrainedTokenizerFast

import openpi.shared.download as download


_FAST_PROCESSOR_CONFIG = "processor_config.json"
_FAST_PROCESSING_MODULE = "processing_action_tokenizer.py"


def _iter_huggingface_hub_dirs() -> list[pathlib.Path]:
    cache_roots: list[pathlib.Path] = []

    if hub_cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache_roots.append(pathlib.Path(hub_cache).expanduser())
    if hf_home := os.environ.get("HF_HOME"):
        cache_roots.append(pathlib.Path(hf_home).expanduser() / "hub")
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        cache_roots.append(pathlib.Path(xdg_cache_home).expanduser() / "huggingface" / "hub")

    cache_roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")

    for parent in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        cache_roots.append(parent / ".cache" / "huggingface" / "hub")

    unique_roots: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for root in cache_roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        unique_roots.append(resolved_root)

    return unique_roots


def _resolve_snapshot_dir(path_or_repo_id: str) -> pathlib.Path | None:
    path = pathlib.Path(path_or_repo_id).expanduser()
    if path.is_dir():
        if (path / _FAST_PROCESSOR_CONFIG).is_file():
            return path.resolve()
        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            refs_main = path / "refs" / "main"
            if refs_main.is_file():
                revision = refs_main.read_text().strip()
                candidate = snapshots_dir / revision
                if (candidate / _FAST_PROCESSOR_CONFIG).is_file():
                    return candidate.resolve()
            for candidate in sorted(snapshots_dir.iterdir()):
                if candidate.is_dir() and (candidate / _FAST_PROCESSOR_CONFIG).is_file():
                    return candidate.resolve()
        return None

    try:
        cached_processor = try_to_load_from_cache(path_or_repo_id, _FAST_PROCESSOR_CONFIG)
    except HFValidationError:
        if path.is_absolute() or path_or_repo_id.startswith("."):
            return None
        raise
    if isinstance(cached_processor, str):
        return pathlib.Path(cached_processor).expanduser().parent.resolve()

    model_cache_dir_name = f"models--{path_or_repo_id.replace('/', '--')}"
    for hub_dir in _iter_huggingface_hub_dirs():
        candidate = hub_dir / model_cache_dir_name
        if not candidate.exists():
            continue
        resolved_candidate = _resolve_snapshot_dir(str(candidate))
        if resolved_candidate is not None:
            return resolved_candidate

    return None


def _load_fast_processor_from_local_dir(tokenizer_dir: pathlib.Path):
    module_path = tokenizer_dir / _FAST_PROCESSING_MODULE
    if not module_path.is_file():
        raise FileNotFoundError(f"FAST processor module not found at {module_path}")
    config_path = tokenizer_dir / _FAST_PROCESSOR_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"FAST processor config not found at {config_path}")

    # Load from source so a stale/corrupt __pycache__ entry in the HF snapshot
    # cannot shadow the real processor implementation.
    module = types.ModuleType("openpi_fast_processing_action_tokenizer")
    module.__file__ = str(module_path)
    exec(compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec"), module.__dict__)
    processor_class = getattr(module, "UniversalActionProcessor", None)
    if processor_class is None:
        raise AttributeError(f"UniversalActionProcessor is missing from {module_path}")

    with config_path.open(encoding="utf-8") as f:
        processor_config = json.load(f)

    return processor_class(
        PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True),
        scale=processor_config["scale"],
        vocab_size=processor_config["vocab_size"],
        min_token=processor_config["min_token"],
        action_dim=processor_config.get("action_dim"),
        time_horizon=processor_config.get("time_horizon"),
    )


class _LazyFASTProcessor:
    """Pickle-safe wrapper around the FAST processor.

    The real processor may come from a dynamically imported module, which is not safe to
    pickle into PyTorch spawn workers. We only materialize that processor lazily inside the
    process that actually needs it.
    """

    def __init__(self, fast_tokenizer_path: str, local_fast_tokenizer_dir: pathlib.Path | None):
        self._fast_tokenizer_path = fast_tokenizer_path
        self._local_fast_tokenizer_dir = None if local_fast_tokenizer_dir is None else str(local_fast_tokenizer_dir)
        self._processor = None

    def __call__(self, *args, **kwargs):
        return self._get_processor()(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._get_processor().decode(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_processor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._processor = None

    def _get_processor(self):
        if self._processor is not None:
            return self._processor

        if self._local_fast_tokenizer_dir is not None:
            tokenizer_dir = pathlib.Path(self._local_fast_tokenizer_dir)
            logging.info(f"Loading FAST tokenizer from local snapshot: {tokenizer_dir}")
            try:
                self._processor = _load_fast_processor_from_local_dir(tokenizer_dir)
                return self._processor
            except Exception as e:
                logging.warning(
                    "Failed to load FAST tokenizer from local snapshot %s, falling back to AutoProcessor: %s",
                    tokenizer_dir,
                    e,
                )

        self._processor = AutoProcessor.from_pretrained(self._fast_tokenizer_path, trust_remote_code=True)
        return self._processor


class PaligemmaTokenizer:

    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently.")
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:

    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        fast_tokenizer_path = os.environ.get("OPENPI_FAST_TOKENIZER_PATH", fast_tokenizer_path)
        local_fast_tokenizer_dir = _resolve_snapshot_dir(fast_tokenizer_path)
        self._fast_tokenizer = _LazyFASTProcessor(fast_tokenizer_path, local_fast_tokenizer_dir)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(self, prompt: str, state: np.ndarray,
                 actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (self._paligemma_tokenizer.encode("Action: ") + action_tokens_in_pg.tolist() +
                              self._paligemma_tokenizer.encode("|"))
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently.")
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=action_horizon,
                                           action_dim=action_dim)[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
