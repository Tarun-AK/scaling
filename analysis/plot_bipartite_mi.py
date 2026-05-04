"""Plot bipartite mutual information I(A:B) vs N.

Usage:
  python analysis/plot_bipartite_mi.py --group <group>
  python analysis/plot_bipartite_mi.py --group <group> --estimator sampled
  python analysis/plot_bipartite_mi.py --group <group> --estimator logged sampled
  python analysis/plot_bipartite_mi.py --group <group> --estimator sampled --hf-model openai-community/gpt2-xl
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import termios
import time
import tty
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

import wandb

WANDB_PROJECT = "tarunadvaith-/scaling"
DEFAULT_N_VALUES = [
    # 2,
    # 4,
    6,
    8,
    10,
    12,
    14,
    16,
    20,
    22,
    26,
    32,
    38,
    44,
    52,
    62,
    74,
    88,
    104,
    122,
    146,
    172,
    206,
    244,
    288,
    342,
    406,
    482,
    572,
    680,
    806,
    958,
    1136,
    1348,
    1600,
]
DEFAULT_MIN_N = min(DEFAULT_N_VALUES)
DEFAULT_MAX_N = 1600
DEFAULT_NUM_N_VALUES = 40
DEFAULT_HF_SAVE_EVERY_BATCHES = 8
FIT_NMAX = 128
plt.style.use("~/plotStyle.mplstyle")


def _show_image(path: str) -> None:
    if shutil.which("kitten") is None:
        return
    subprocess.run(["kitten", "icat", path], check=False)
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        sys.stdin.read(1)
    subprocess.run(["kitten", "icat", "--clear"], check=False)


def _resolve_group_runs(api: wandb.Api, group: str) -> list[wandb.apis.public.Run]:
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={
                "group": group,
                "state": "finished",
            },
        )
    )
    if not runs:
        raise RuntimeError(f"No finished runs found for group='{group}'")

    by_hidden_dim: dict[int, wandb.apis.public.Run] = {}
    for run in runs:
        cfg = run.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        hidden_dim = int(hidden_dim)
        if hidden_dim in by_hidden_dim:
            raise RuntimeError(
                "Multiple finished runs found for "
                f"group='{group}' hidden_dim={hidden_dim}"
            )
        by_hidden_dim[hidden_dim] = run

    if not by_hidden_dim:
        raise RuntimeError(
            f"No finished runs with config.hidden_dim found for group='{group}'"
        )

    return [by_hidden_dim[k] for k in sorted(by_hidden_dim)]


def _wandb_entity_project() -> tuple[str, str]:
    entity, project = WANDB_PROJECT.split("/")
    return entity, project


def _download_checkpoint_artifact(
    run_id: str,
    api: wandb.Api,
    cache_dir: str,
) -> str:
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cached_ckpt = os.path.join(cache_dir, run_id, "ckpt")
    if os.path.exists(cached_ckpt):
        return cached_ckpt

    entity, project = _wandb_entity_project()
    artifact_name = f"{entity}/{project}/checkpoint-{run_id}:latest"
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=os.path.join(cache_dir, run_id))
    return os.path.join(artifact_dir, "ckpt")


def _load_checkpoint(ckpt_path: str, state) -> tuple[Any, dict]:
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"]), restored


def _normalize_params_for_step(params: dict, num_layers: int) -> dict:
    if any(k.startswith("rnn_") for k in params):
        return params

    lstm_keys = [k for k in params if k.startswith("LSTMCell_")]
    if not lstm_keys:
        return params

    normalized = dict(params)
    for layer_idx in range(num_layers):
        cell_key = f"LSTMCell_{layer_idx}"
        if cell_key not in params:
            raise RuntimeError(
                f"Missing {cell_key} in params for num_layers={num_layers}"
            )
        normalized.setdefault(f"rnn_{layer_idx}", {"cell": params[cell_key]})
    return normalized


def _sample_sequences(
    model,
    params: dict,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
    rng,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    def _sample_batch(
        batch_rng,
        init_carry: tuple,
        seq_len: int,
        bos_token_id: int,
    ):
        def scan_step(state, _):
            lstm_carry, logits = model.apply(
                {"params": params},
                state["carry"],
                state["token"],
                method=model.step,
            )
            rng, step_rng = jax.random.split(state["rng"])
            next_token = jax.random.categorical(step_rng, logits)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_logp = jnp.take_along_axis(
                log_probs,
                next_token[:, None],
                axis=-1,
            ).squeeze(-1)
            new_state = {"carry": lstm_carry, "token": next_token, "rng": rng}
            return new_state, (next_token, token_logp)

        init_token = jnp.full((init_carry[0][0].shape[0],), bos_token_id)
        carry = {"carry": init_carry, "token": init_token, "rng": batch_rng}
        _, (tokens, logps) = jax.lax.scan(scan_step, carry, jnp.arange(seq_len))
        return jnp.transpose(tokens, (1, 0)), jnp.transpose(logps, (1, 0))

    sample_batch_jit = jax.jit(
        _sample_batch,
        static_argnames=("seq_len", "bos_token_id"),
    )

    all_samples = []
    all_logps = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(num_batches),
        desc=progress_desc,
        unit="batch",
        leave=False,
    ):
        rng, batch_rng = jax.random.split(rng)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        init_carry = model.init_carry(current_batch_size)
        tokens, logps = sample_batch_jit(batch_rng, init_carry, seq_len, bos_token_id)
        all_samples.append(np.array(tokens))
        all_logps.append(np.array(logps))

    samples = np.concatenate(all_samples, axis=0)[:num_samples]
    sample_logps = np.concatenate(all_logps, axis=0)[:num_samples]
    return samples, sample_logps


def _prepend_bos(tokens, bos_token_id: int):
    import jax.numpy as jnp

    bos = jnp.full((tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


def _score_logps(
    apply_fn,
    params: dict,
    tokens,
    batch_size: int,
    bos_token_id: int,
    progress_desc: str,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    tokens_device = jax.device_put(tokens)
    num_tokens = int(tokens_device.shape[0])
    if num_tokens == 0:
        seq_len = int(tokens_device.shape[1])
        return np.empty((0, seq_len), dtype=np.float32)

    fixed_batch_size = min(batch_size, num_tokens)

    @jax.jit
    def _score_batch(batch_tokens: jax.Array, batch_params: dict) -> jax.Array:
        inputs = _prepend_bos(batch_tokens, bos_token_id)
        logits = apply_fn({"params": batch_params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(
            log_probs,
            batch_tokens[:, :, None],
            axis=-1,
        ).squeeze(-1)

    all_logps = []
    num_batches = (num_tokens + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(num_batches),
        desc=progress_desc,
        unit="batch",
        leave=False,
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_tokens)
        current_size = end - start
        batch = tokens_device[start:end]
        if current_size < fixed_batch_size:
            pad_rows = fixed_batch_size - current_size
            pad = jnp.repeat(batch[:1], pad_rows, axis=0)
            batch = jnp.concatenate([batch, pad], axis=0)

        token_logp = _score_batch(batch, params)
        all_logps.append(np.array(token_logp[:current_size]))

    return np.concatenate(all_logps, axis=0)[:num_tokens]


def _sample_cache_key(
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
) -> str:
    return (
        f"seq{int(seq_len)}_num{int(num_samples)}_"
        f"samplebs{int(batch_size)}_bos{int(bos_token_id)}"
    )


def _sample_cache_paths(
    cache_dir: str, run_id: str, sample_key: str
) -> tuple[str, str]:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    os.makedirs(run_cache_dir, exist_ok=True)
    sample_cache_path = os.path.join(run_cache_dir, f"samples_{sample_key}.npz")
    log_q_y_cache_path = os.path.join(run_cache_dir, f"log_q_y_{sample_key}.npz")
    return sample_cache_path, log_q_y_cache_path


def _sample_key_from_sample_cache_path(sample_cache_path: str) -> str | None:
    filename = os.path.basename(sample_cache_path)
    if not filename.startswith("samples_") or not filename.endswith(".npz"):
        return None
    return filename[len("samples_") : -len(".npz")]


def _log_q_y_cache_path_for_sample_cache(sample_cache_path: str) -> str:
    sample_key = _sample_key_from_sample_cache_path(sample_cache_path)
    if sample_key is None:
        raise RuntimeError(f"Invalid sample cache filename: {sample_cache_path}")
    return os.path.join(
        os.path.dirname(sample_cache_path),
        f"log_q_y_{sample_key}.npz",
    )


def _compatible_sample_cache_entries(
    cache_dir: str,
    run_id: str,
    seq_len: int,
    batch_size: int | None,
    bos_token_id: int,
) -> list[tuple[int, str, str]]:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    if not os.path.isdir(run_cache_dir):
        return []

    pattern = re.compile(
        r"^samples_seq(?P<seq>\d+)_num(?P<num>\d+)_"
        rf"samplebs(?P<samplebs>\d+)_bos{int(bos_token_id)}\.npz$"
    )
    entries: list[tuple[int, str, str]] = []
    for filename in os.listdir(run_cache_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        seq = int(match.group("seq"))
        if seq < int(seq_len):
            continue
        samplebs = int(match.group("samplebs"))
        if batch_size is not None and samplebs != int(batch_size):
            continue
        num_samples = int(match.group("num"))
        if num_samples < 1:
            continue
        sample_cache_path = os.path.join(run_cache_dir, filename)
        log_q_y_cache_path = _log_q_y_cache_path_for_sample_cache(sample_cache_path)
        entries.append((num_samples, sample_cache_path, log_q_y_cache_path))
    entries.sort(key=lambda item: item[0])
    return entries


def _find_reusable_complete_cache(
    cache_dir: str,
    run_id: str,
    seq_len: int,
    target_num_samples: int,
    batch_size: int,
    bos_token_id: int,
    n_values: list[int],
) -> tuple[np.ndarray, dict[int, float], int, str, str] | None:
    entries = _compatible_sample_cache_entries(
        cache_dir=cache_dir,
        run_id=run_id,
        seq_len=seq_len,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
    )
    if not entries:
        entries = _compatible_sample_cache_entries(
            cache_dir=cache_dir,
            run_id=run_id,
            seq_len=seq_len,
            batch_size=None,
            bos_token_id=bos_token_id,
        )
    best_ge_target: tuple[np.ndarray, dict[int, float], int, str, str] | None = None
    best_any: tuple[np.ndarray, dict[int, float], int, str, str] | None = None
    for parsed_num_samples, sample_cache_path, log_q_y_cache_path in entries:
        log_q_y_means_by_n = _load_log_q_y_mean_cache(log_q_y_cache_path)
        sample_logps = _load_sample_logps(sample_cache_path)
        if sample_logps is None:
            continue
        actual_num_samples = int(sample_logps.shape[0])
        if actual_num_samples < 1:
            continue
        available_n_values = _select_cached_n_values(
            n_values,
            log_q_y_means_by_n,
            sample_logps,
        )
        if not available_n_values:
            continue
        candidate = (
            sample_logps,
            log_q_y_means_by_n,
            actual_num_samples,
            sample_cache_path,
            log_q_y_cache_path,
        )
        if best_any is None or actual_num_samples > best_any[2]:
            best_any = candidate
        if actual_num_samples >= int(target_num_samples):
            if best_ge_target is None or actual_num_samples > best_ge_target[2]:
                best_ge_target = candidate
    if best_ge_target is not None:
        return best_ge_target
    return best_any


def _incremental_sample_cache_path(
    sample_cache_path: str,
    existing_num_samples: int,
    target_num_samples: int,
) -> str:
    root, ext = os.path.splitext(sample_cache_path)
    return (
        f"{root}_from{int(existing_num_samples)}_to{int(target_num_samples)}"
        f"{ext or '.npz'}"
    )


def _hf_cache_id(model_name: str, revision: str | None) -> str:
    payload = f"{model_name}@{revision or 'main'}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model_name)
    safe_revision = (
        "main"
        if revision is None
        else "".join(ch if ch.isalnum() else "_" for ch in revision)
    )
    return f"hf_{safe_model}_{safe_revision}_{digest}"


def _resolve_hf_bos_and_max_positions(
    model_name: str,
    revision: str | None,
) -> tuple[int, int]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    max_positions = getattr(config, "n_positions", None)
    if max_positions is None:
        max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None:
        max_positions = getattr(config, "n_ctx", None)
    if max_positions is None:
        raise RuntimeError(
            "Failed to determine context length from HF config for "
            f"model='{model_name}'"
        )

    bos_token_id = getattr(config, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(config, "eos_token_id", None)
    if bos_token_id is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise RuntimeError(
            "Failed to determine bos_token_id for HF model " f"'{model_name}'"
        )

    return int(bos_token_id), int(max_positions)


def _is_torch_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


def _is_hf_compile_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    indicators = (
        "inductorerror",
        "torch._inductor",
        "torch._dynamo",
        "triton",
        "cudagraph",
        "python.h",
        "cuda_utils.c",
    )
    return any(indicator in message for indicator in indicators)


def _mark_cudagraph_step_begin(torch) -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is None:
        return
    marker()


def _torch_autocast_context(torch, device, dtype):
    if device.type != "cuda" or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _hf_partial_sample_paths(sample_cache_path: str) -> tuple[str, str, str, str]:
    partial_dir = f"{sample_cache_path}.partial"
    samples_mm_path = os.path.join(partial_dir, "samples.memmap")
    sample_logps_mm_path = os.path.join(partial_dir, "sample_logps.memmap")
    meta_path = os.path.join(partial_dir, "meta.npz")
    return partial_dir, samples_mm_path, sample_logps_mm_path, meta_path


def _save_hf_partial_meta(
    meta_path: str,
    num_samples: int,
    seq_len: int,
    completed: int,
) -> None:
    tmp_meta_path = f"{meta_path}.tmp.npz"
    np.savez(
        tmp_meta_path,
        num_samples=np.array(int(num_samples), dtype=np.int64),
        seq_len=np.array(int(seq_len), dtype=np.int64),
        completed=np.array(int(completed), dtype=np.int64),
    )
    os.replace(tmp_meta_path, meta_path)


def _load_or_init_hf_partial_samples(
    sample_cache_path: str,
    num_samples: int,
    seq_len: int,
) -> tuple[np.memmap, np.memmap, int, str, str]:
    partial_dir, samples_mm_path, sample_logps_mm_path, meta_path = (
        _hf_partial_sample_paths(
            sample_cache_path,
        )
    )
    os.makedirs(partial_dir, exist_ok=True)

    expected_samples_bytes = (
        int(num_samples) * int(seq_len) * np.dtype(np.int32).itemsize
    )
    expected_logps_bytes = (
        int(num_samples) * int(seq_len) * np.dtype(np.float32).itemsize
    )

    resume = (
        os.path.exists(meta_path)
        and os.path.exists(samples_mm_path)
        and os.path.exists(sample_logps_mm_path)
        and os.path.getsize(samples_mm_path) == expected_samples_bytes
        and os.path.getsize(sample_logps_mm_path) == expected_logps_bytes
    )

    completed = 0
    if resume:
        try:
            with np.load(meta_path) as meta:
                meta_num_samples = int(meta.get("num_samples", -1))
                meta_seq_len = int(meta.get("seq_len", -1))
                meta_completed = int(meta.get("completed", 0))
            if meta_num_samples != int(num_samples) or meta_seq_len != int(seq_len):
                resume = False
            else:
                completed = max(0, min(int(num_samples), meta_completed))
        except Exception:
            resume = False

    if not resume:
        for path in (meta_path, samples_mm_path, sample_logps_mm_path):
            if os.path.exists(path):
                os.remove(path)

    mode = "r+" if resume else "w+"
    samples_mm = np.memmap(
        samples_mm_path,
        dtype=np.int32,
        mode=mode,
        shape=(int(num_samples), int(seq_len)),
    )
    sample_logps_mm = np.memmap(
        sample_logps_mm_path,
        dtype=np.float32,
        mode=mode,
        shape=(int(num_samples), int(seq_len)),
    )
    if not resume:
        _save_hf_partial_meta(meta_path, num_samples, seq_len, completed=0)
    else:
        print(
            "Resuming partial HF samples: "
            f"{completed}/{num_samples} sequences already cached"
        )

    return samples_mm, sample_logps_mm, completed, partial_dir, meta_path


def _cleanup_hf_partial_samples(sample_cache_path: str) -> None:
    partial_dir, *_ = _hf_partial_sample_paths(sample_cache_path)
    if os.path.exists(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)


def _maybe_compile_hf_model(
    model,
    torch,
    compile_mode: str,
    *,
    enabled: bool,
    compile_for: str,
):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; continuing without compilation")
        return model
    try:
        compiled_model = torch.compile(
            model,
            mode=compile_mode,
            options={"triton.cudagraphs": False},
        )
        print(
            f"Enabled torch.compile for HF {compile_for} path "
            f"(mode={compile_mode}, cudagraphs_disabled=True)"
        )
        return compiled_model
    except Exception as exc:
        try:
            compiled_model = torch.compile(model, mode=compile_mode)
            print(
                f"Enabled torch.compile for HF {compile_for} path "
                f"(mode={compile_mode}, default_options)"
            )
            return compiled_model
        except Exception as exc_fallback:
            print(
                f"torch.compile failed for HF {compile_for} path; "
                f"continuing without compile. "
                f"Error with cudagraphs disabled: {exc}; "
                f"error with default options: {exc_fallback}"
            )
            return model


def _load_hf_torch_model(
    model_name: str,
    revision: str | None,
    attn_implementation: str | None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --hf-model. Install it in .venv first."
        ) from exc

    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type == "cpu":
        print(
            "Warning: torch.cuda.is_available() is False; HF sampling/scoring "
            "will run on CPU and be very slow."
        )

    if device.type == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": dtype,
    }
    if device.type == "cuda" and attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        if "attn_implementation" not in str(exc):
            raise
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.to(device)
    model.eval()

    param_dtype = None
    try:
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = dtype

    autocast_dtype = param_dtype if device.type == "cuda" else None
    print(
        "HF model loaded on "
        f"device={device}, param_dtype={param_dtype}, "
        f"attn_impl={attn_implementation or 'default'}"
    )
    return model, torch, device, autocast_dtype


def _sample_single_hf_batch(
    model,
    torch,
    device,
    seq_len: int,
    batch_size: int,
    bos_token_id: int,
    autocast_dtype,
    token_progress,
    token_log_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )
    tokens = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    token_logps = torch.empty((batch_size, seq_len), dtype=torch.float32, device=device)
    past_key_values = None
    pending_steps = 0
    with torch.inference_mode():
        with _torch_autocast_context(torch, device, autocast_dtype):
            for step in range(seq_len):
                _mark_cudagraph_step_begin(torch)
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                logp = torch.gather(log_probs, 1, next_token[:, None]).squeeze(1)

                tokens[:, step] = next_token
                token_logps[:, step] = logp
                input_ids = next_token[:, None]

                pending_steps += 1
                if token_progress is not None and (
                    (step + 1) % token_log_every == 0 or (step + 1) == seq_len
                ):
                    token_progress.update(pending_steps)
                    pending_steps = 0

    return (
        tokens.cpu().numpy().astype(np.int32, copy=False),
        token_logps.cpu().numpy().astype(np.float32, copy=False),
    )


def _sample_sequences_hf(
    model,
    torch,
    device,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
    seed: int,
    progress_desc: str,
    sample_cache_path: str,
    autocast_dtype,
    save_every_batches: int = DEFAULT_HF_SAVE_EVERY_BATCHES,
    token_log_every: int = 1,
    sample_progress: Any | None = None,
    show_token_progress: bool = True,
    on_sampled_batch: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if save_every_batches < 1:
        raise RuntimeError("save_every_batches must be >= 1")
    if token_log_every < 1:
        raise RuntimeError("token_log_every must be >= 1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples_mm, sample_logps_mm, completed, _, meta_path = (
        _load_or_init_hf_partial_samples(
            sample_cache_path=sample_cache_path,
            num_samples=num_samples,
            seq_len=seq_len,
        )
    )

    active_batch_size = max(1, int(batch_size))
    batch_counter = 0
    owns_progress = sample_progress is None
    if sample_progress is None:
        sample_progress = tqdm(
            total=int(num_samples),
            initial=int(completed),
            desc=progress_desc,
            unit="sample",
            leave=False,
        )

    try:
        while completed < num_samples:
            current_batch_size = min(active_batch_size, num_samples - completed)
            start_time = time.perf_counter()
            token_progress = (
                tqdm(
                    total=seq_len,
                    desc=f"{progress_desc} token-steps",
                    unit="tok",
                    leave=False,
                )
                if show_token_progress
                else None
            )
            try:
                batch_tokens, batch_logps = _sample_single_hf_batch(
                    model=model,
                    torch=torch,
                    device=device,
                    seq_len=seq_len,
                    batch_size=current_batch_size,
                    bos_token_id=bos_token_id,
                    autocast_dtype=autocast_dtype,
                    token_progress=token_progress,
                    token_log_every=token_log_every,
                )
            except RuntimeError as exc:
                if not _is_torch_oom_error(exc) or current_batch_size <= 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    "HF sampling OOM at batch_size="
                    f"{current_batch_size}; retrying with batch_size={new_batch_size}"
                )
                active_batch_size = new_batch_size
                continue
            finally:
                if token_progress is not None:
                    token_progress.close()

            if on_sampled_batch is not None:
                on_sampled_batch(batch_tokens, batch_logps)

            end = completed + current_batch_size
            samples_mm[completed:end] = batch_tokens
            sample_logps_mm[completed:end] = batch_logps
            completed = end
            batch_counter += 1
            sample_progress.update(current_batch_size)

            elapsed = max(time.perf_counter() - start_time, 1e-8)
            tokens_per_second = float(current_batch_size * seq_len) / elapsed
            samples_per_second = float(current_batch_size) / elapsed
            sample_progress.set_postfix(
                sample_bs=current_batch_size,
                sample_s=f"{samples_per_second:,.2f}",
                tok_s=f"{tokens_per_second:,.0f}",
            )

            if batch_counter % save_every_batches == 0 or completed == num_samples:
                samples_mm.flush()
                sample_logps_mm.flush()
                _save_hf_partial_meta(meta_path, num_samples, seq_len, completed)
    finally:
        if owns_progress:
            sample_progress.close()

    samples = np.array(samples_mm, copy=True)
    sample_logps = np.array(sample_logps_mm, copy=True)
    del samples_mm
    del sample_logps_mm

    _save_sample_cache(sample_cache_path, samples, sample_logps)
    _cleanup_hf_partial_samples(sample_cache_path)
    return samples, sample_logps


def _score_logps_hf(
    model,
    torch,
    device,
    tokens: np.ndarray,
    batch_size: int,
    bos_token_id: int,
    progress_desc: str | None,
    autocast_dtype,
    show_progress: bool = True,
) -> np.ndarray:
    num_tokens = int(tokens.shape[0])
    if num_tokens == 0:
        seq_len = int(tokens.shape[1])
        return np.empty((0, seq_len), dtype=np.float32)

    seq_len = int(tokens.shape[1])
    all_logps = np.empty((num_tokens, seq_len), dtype=np.float32)
    active_batch_size = max(1, min(int(batch_size), num_tokens))
    start = 0

    use_progress = bool(show_progress and progress_desc is not None)
    sample_progress = (
        tqdm(
            total=num_tokens,
            desc=progress_desc,
            unit="sample",
            leave=False,
        )
        if use_progress
        else None
    )
    try:
        while start < num_tokens:
            current_batch_size = min(active_batch_size, num_tokens - start)
            batch_np = tokens[start : start + current_batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            bos = torch.full(
                (current_batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device,
            )
            inputs = torch.cat([bos, batch[:, :-1]], dim=1)

            step_start = time.perf_counter()
            try:
                with torch.inference_mode():
                    with _torch_autocast_context(torch, device, autocast_dtype):
                        _mark_cudagraph_step_begin(torch)
                        logits = model(input_ids=inputs, use_cache=False).logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_logp = torch.gather(
                        log_probs,
                        2,
                        batch.unsqueeze(-1),
                    ).squeeze(-1)
            except RuntimeError as exc:
                if not _is_torch_oom_error(exc) or current_batch_size <= 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    "HF scoring OOM at batch_size="
                    f"{current_batch_size}; retrying with batch_size={new_batch_size}"
                )
                active_batch_size = new_batch_size
                continue

            all_logps[start : start + current_batch_size] = (
                token_logp.cpu().numpy().astype(np.float32, copy=False)
            )
            start += current_batch_size

            elapsed = max(time.perf_counter() - step_start, 1e-8)
            tokens_per_second = float(current_batch_size * seq_len) / elapsed
            if sample_progress is not None:
                sample_progress.update(current_batch_size)
                sample_progress.set_postfix(
                    score_bs=current_batch_size,
                    tok_s=f"{tokens_per_second:,.0f}",
                )
    finally:
        if sample_progress is not None:
            sample_progress.close()

    return all_logps


def _accumulate_log_q_y_sums_hf(
    tokens: np.ndarray,
    n_values: list[int],
    model,
    torch,
    device,
    batch_size: int,
    bos_token_id: int,
    autocast_dtype,
    sums_by_n: dict[int, float],
) -> None:
    for n in n_values:
        half = n // 2
        y_tokens = tokens[:, half:n]
        y_logps = _score_logps_hf(
            model=model,
            torch=torch,
            device=device,
            tokens=y_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=None,
            autocast_dtype=autocast_dtype,
            show_progress=False,
        )
        sums_by_n[n] += float(np.sum(y_logps, dtype=np.float64))


def _sample_and_score_hf_in_succession(
    sample_model,
    score_model,
    torch,
    device,
    seq_len: int,
    target_num_samples: int,
    n_values: list[int],
    sample_batch_size: int,
    score_batch_size: int,
    bos_token_id: int,
    seed: int,
    sample_cache_path: str,
    autocast_dtype,
    save_every_batches: int,
    token_log_every: int,
    progress_desc: str,
    existing_samples: np.ndarray | None = None,
    existing_sample_logps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    cached_count = 0
    if existing_samples is not None and existing_sample_logps is not None:
        cached_count = min(
            int(existing_samples.shape[0]), int(existing_sample_logps.shape[0])
        )
        if cached_count > 0:
            existing_samples = existing_samples[:cached_count]
            existing_sample_logps = existing_sample_logps[:cached_count]

    if cached_count > int(target_num_samples):
        cached_count = int(target_num_samples)
        if existing_samples is not None and existing_sample_logps is not None:
            existing_samples = existing_samples[:cached_count]
            existing_sample_logps = existing_sample_logps[:cached_count]

    missing_sample_count = max(0, int(target_num_samples) - cached_count)
    sums_by_n = {int(n): 0.0 for n in n_values}

    progress = tqdm(
        total=int(target_num_samples),
        desc=progress_desc,
        unit="sample",
        leave=False,
    )
    try:
        if cached_count > 0 and existing_samples is not None:
            start = 0
            while start < cached_count:
                end = min(start + int(score_batch_size), cached_count)
                batch_tokens = existing_samples[start:end]
                _accumulate_log_q_y_sums_hf(
                    tokens=batch_tokens,
                    n_values=n_values,
                    model=score_model,
                    torch=torch,
                    device=device,
                    batch_size=score_batch_size,
                    bos_token_id=bos_token_id,
                    autocast_dtype=autocast_dtype,
                    sums_by_n=sums_by_n,
                )
                batch_size = end - start
                progress.update(batch_size)
                progress.set_postfix(stage="score", bs=batch_size)
                start = end

        new_samples = np.empty((0, int(seq_len)), dtype=np.int32)
        new_sample_logps = np.empty((0, int(seq_len)), dtype=np.float32)
        if missing_sample_count > 0:
            incremental_sample_cache_path = _incremental_sample_cache_path(
                sample_cache_path=sample_cache_path,
                existing_num_samples=cached_count,
                target_num_samples=target_num_samples,
            )
            if os.path.exists(incremental_sample_cache_path):
                os.remove(incremental_sample_cache_path)
            _cleanup_hf_partial_samples(incremental_sample_cache_path)

            def _on_sampled_batch(
                batch_tokens: np.ndarray, batch_logps: np.ndarray
            ) -> None:
                del batch_logps
                _accumulate_log_q_y_sums_hf(
                    tokens=batch_tokens,
                    n_values=n_values,
                    model=score_model,
                    torch=torch,
                    device=device,
                    batch_size=score_batch_size,
                    bos_token_id=bos_token_id,
                    autocast_dtype=autocast_dtype,
                    sums_by_n=sums_by_n,
                )

            new_samples, new_sample_logps = _sample_sequences_hf(
                model=sample_model,
                torch=torch,
                device=device,
                seq_len=seq_len,
                num_samples=missing_sample_count,
                batch_size=sample_batch_size,
                bos_token_id=bos_token_id,
                seed=int(seed) + int(cached_count),
                progress_desc=progress_desc,
                sample_cache_path=incremental_sample_cache_path,
                autocast_dtype=autocast_dtype,
                save_every_batches=save_every_batches,
                token_log_every=token_log_every,
                sample_progress=progress,
                show_token_progress=False,
                on_sampled_batch=_on_sampled_batch,
            )
            progress.set_postfix(stage="sample+score", bs=int(sample_batch_size))

        if (
            existing_samples is None
            or existing_sample_logps is None
            or cached_count == 0
        ):
            samples = new_samples
            sample_logps = new_sample_logps
        else:
            samples = np.concatenate([existing_samples, new_samples], axis=0)
            sample_logps = np.concatenate(
                [existing_sample_logps, new_sample_logps], axis=0
            )

        _save_sample_cache(sample_cache_path, samples, sample_logps)
    finally:
        progress.close()

    if samples.shape[0] == 0:
        raise RuntimeError("No HF samples available after sampling/scoring")

    denominator = float(samples.shape[0])
    log_q_y_means_by_n = {
        int(n): float(sums_by_n[int(n)] / denominator) for n in n_values
    }
    return samples, sample_logps, log_q_y_means_by_n


def _load_sample_cache(sample_cache_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(sample_cache_path):
        return None
    with np.load(sample_cache_path) as data:
        if "samples" not in data or "sample_logps" not in data:
            return None
        samples = np.array(data["samples"])
        sample_logps = np.array(data["sample_logps"])
    if samples.shape != sample_logps.shape:
        return None
    return samples, sample_logps


def _load_sample_logps(sample_cache_path: str) -> np.ndarray | None:
    if not os.path.exists(sample_cache_path):
        return None
    with np.load(sample_cache_path) as data:
        if "sample_logps" not in data:
            return None
        sample_logps = np.array(data["sample_logps"])
    if sample_logps.ndim != 2:
        return None
    return sample_logps


def _save_sample_cache(
    sample_cache_path: str,
    samples: np.ndarray,
    sample_logps: np.ndarray,
) -> None:
    np.savez_compressed(
        sample_cache_path,
        samples=samples,
        sample_logps=sample_logps,
    )


def _load_log_q_y_mean_cache(log_q_y_cache_path: str) -> dict[int, float]:
    if not os.path.exists(log_q_y_cache_path):
        return {}
    with np.load(log_q_y_cache_path) as data:
        if "n_values" not in data or "log_q_y_means" not in data:
            return {}
        n_values = np.array(data["n_values"])
        log_q_y_means = np.array(data["log_q_y_means"])
    if len(n_values) != len(log_q_y_means):
        return {}
    out: dict[int, float] = {}
    for n, value in zip(n_values, log_q_y_means):
        n_int = int(n)
        value_float = float(value)
        if np.isfinite(value_float):
            out[n_int] = value_float
    return out


def _save_log_q_y_mean_cache(
    log_q_y_cache_path: str,
    log_q_y_means_by_n: dict[int, float],
) -> None:
    n_values = np.array(sorted(log_q_y_means_by_n.keys()), dtype=np.int32)
    log_q_y_means = np.array(
        [log_q_y_means_by_n[int(n)] for n in n_values],
        dtype=np.float32,
    )
    np.savez_compressed(
        log_q_y_cache_path,
        n_values=n_values,
        log_q_y_means=log_q_y_means,
    )


def _compute_log_q_y_means(
    samples,
    apply_fn,
    params: dict,
    n_values: list[int],
    batch_size: int,
    bos_token_id: int,
    progress_desc_prefix: str,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for n in n_values:
        half = n // 2
        y_tokens = samples[:, half:n]
        y_logps = _score_logps(
            apply_fn=apply_fn,
            params=params,
            tokens=y_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"{progress_desc_prefix} N={n}",
        )
        out[n] = float(np.mean(np.sum(y_logps, axis=1)))
    return out


def _extract_conditional_entropy(run: wandb.apis.public.Run) -> dict[int, float]:
    out: dict[int, float] = {}

    summary = run.summary or {}
    for key, value in summary.items():
        if not str(key).startswith("conditional_entropy/entropy_"):
            continue
        try:
            n = int(str(key).rsplit("_", 1)[1])
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val

    if out:
        return out

    history = run.history()
    if history is None or history.empty:
        return out
    keys = [k for k in history.columns if k.startswith("conditional_entropy/entropy_")]
    if not keys:
        return out

    valid = history[keys].dropna(how="all")
    if valid.empty:
        return out

    last_row = valid.iloc[-1]
    for key in keys:
        value = last_row.get(key)
        try:
            n = int(str(key).rsplit("_", 1)[1])
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val
    return out


def _compute_bipartite_mi_from_conditional_entropy(
    conditional_entropy: dict[int, float],
    n_values: list[int],
) -> dict[int, float]:
    if not conditional_entropy:
        return {}

    max_n = max(conditional_entropy)
    entropy = np.array(
        [conditional_entropy.get(i, np.nan) for i in range(1, max_n + 1)],
        dtype=float,
    )
    cumsum = np.cumsum(entropy)

    out: dict[int, float] = {}
    for n in n_values:
        if n < 1 or n > max_n:
            continue
        half = n // 2
        if half < 1:
            continue
        s_ab = float(cumsum[n - 1])
        s_a = float(cumsum[half - 1])
        if np.isfinite(s_ab) and np.isfinite(s_a):
            out[n] = 2.0 * s_a - s_ab
    return out


def _compute_bipartite_mi_from_sampled_q(
    sample_logps: np.ndarray,
    n_values: list[int],
    log_q_y_means_by_n: dict[int, float],
) -> dict[int, float]:
    sample_logps_cumsum = np.cumsum(sample_logps, axis=1)

    out: dict[int, float] = {}
    for n in n_values:
        if n not in log_q_y_means_by_n:
            raise RuntimeError(f"Missing cached log q(y) for N={n}")
        half = n // 2
        log_q_y_given_x = np.mean(
            sample_logps_cumsum[:, n - 1] - sample_logps_cumsum[:, half - 1]
        )
        out[n] = float(log_q_y_given_x - log_q_y_means_by_n[n])
    return out


def _compute_hf_sampled_mi_series(
    model_name: str,
    revision: str | None,
    n_values: list[int],
    num_samples: int,
    sample_batch_size: int,
    score_batch_size: int,
    cache_dir: str,
    seed: int,
    save_every_batches: int,
    token_log_every: int,
    compile_target: str,
    compile_mode: str,
    attn_implementation: str | None,
    force_resample: bool = False,
) -> tuple[dict[int, float], int]:
    bos_token_id, max_positions = _resolve_hf_bos_and_max_positions(
        model_name,
        revision,
    )

    capped_n_values = [n for n in n_values if n <= max_positions]
    if not capped_n_values:
        raise RuntimeError(
            f"No N values are <= context limit ({max_positions}) for "
            f"HF model '{model_name}'"
        )

    sample_key = _sample_cache_key(
        seq_len=capped_n_values[-1],
        num_samples=num_samples,
        batch_size=sample_batch_size,
        bos_token_id=bos_token_id,
    )
    hf_cache_id = _hf_cache_id(model_name, revision)
    sample_cache_path, log_q_y_cache_path = _sample_cache_paths(
        cache_dir,
        hf_cache_id,
        sample_key,
    )

    if not force_resample:
        reusable_cache = _find_reusable_complete_cache(
            cache_dir=cache_dir,
            run_id=hf_cache_id,
            seq_len=capped_n_values[-1],
            target_num_samples=num_samples,
            batch_size=sample_batch_size,
            bos_token_id=bos_token_id,
            n_values=capped_n_values,
        )
        if reusable_cache is None:
            raise RuntimeError(
                "No complete cached HF sampled MI found for "
                f"model='{model_name}'. Re-run with --force-resample to regenerate."
            )
        (
            reusable_sample_logps,
            reusable_log_q_y_means_by_n,
            reusable_num_samples,
            reusable_sample_cache_path,
            _,
        ) = reusable_cache
        available_n_values = _select_cached_n_values(
            capped_n_values,
            reusable_log_q_y_means_by_n,
            reusable_sample_logps,
        )
        if not available_n_values:
            raise RuntimeError(
                "Cached HF sampled artifacts have no usable N values for plotting. "
                "Re-run with --force-resample to regenerate."
            )
        if len(available_n_values) < len(capped_n_values):
            print(
                "Using cached HF sampled MI subset: "
                f"{len(available_n_values)}/{len(capped_n_values)} N values available"
            )
        print(
            "Using cached sampled MI for hf_model="
            f"{model_name} from {os.path.basename(reusable_sample_cache_path)} "
            f"(num_samples={reusable_num_samples}, requested={num_samples})"
        )
        mi_values = _compute_bipartite_mi_from_sampled_q(
            sample_logps=reusable_sample_logps,
            n_values=available_n_values,
            log_q_y_means_by_n=reusable_log_q_y_means_by_n,
        )
        return mi_values, max_positions

    print(
        "Force resample enabled for hf_model="
        f"{model_name}; regenerating sampled cache"
    )
    if os.path.exists(sample_cache_path):
        os.remove(sample_cache_path)
    if os.path.exists(log_q_y_cache_path):
        os.remove(log_q_y_cache_path)
    _cleanup_hf_partial_samples(sample_cache_path)

    model = None
    torch_module = None
    device = None
    autocast_dtype = None
    eager_model = None
    sample_model = None
    score_model = None
    sample_compiled = False
    score_compiled = False
    samples: np.ndarray | None = None
    sample_logps: np.ndarray | None = None
    log_q_y_means_by_n: dict[int, float] = {}
    try:
        model, torch_module, device, autocast_dtype = _load_hf_torch_model(
            model_name,
            revision,
            attn_implementation,
        )
        eager_model = model

        sample_model = eager_model
        if compile_target in {"sample", "both"}:
            sample_model = _maybe_compile_hf_model(
                eager_model,
                torch_module,
                compile_mode,
                enabled=True,
                compile_for="sampling",
            )
            sample_compiled = sample_model is not eager_model

        score_model = eager_model
        if compile_target in {"score", "both"}:
            score_model = _maybe_compile_hf_model(
                eager_model,
                torch_module,
                compile_mode,
                enabled=True,
                compile_for="scoring",
            )
            score_compiled = score_model is not eager_model

        try:
            samples, sample_logps, log_q_y_means_by_n = (
                _sample_and_score_hf_in_succession(
                    sample_model=sample_model,
                    score_model=score_model,
                    torch=torch_module,
                    device=device,
                    seq_len=capped_n_values[-1],
                    target_num_samples=num_samples,
                    n_values=capped_n_values,
                    sample_batch_size=sample_batch_size,
                    score_batch_size=score_batch_size,
                    bos_token_id=bos_token_id,
                    seed=seed,
                    sample_cache_path=sample_cache_path,
                    autocast_dtype=autocast_dtype,
                    save_every_batches=save_every_batches,
                    token_log_every=token_log_every,
                    progress_desc=f"Sampling+scoring hf model={model_name}",
                )
            )
        except Exception as exc:
            if (sample_compiled or score_compiled) and _is_hf_compile_runtime_error(
                exc
            ):
                print(
                    "HF sampling/scoring compile backend failed at runtime; "
                    "retrying in eager mode"
                )
                samples, sample_logps, log_q_y_means_by_n = (
                    _sample_and_score_hf_in_succession(
                        sample_model=eager_model,
                        score_model=eager_model,
                        torch=torch_module,
                        device=device,
                        seq_len=capped_n_values[-1],
                        target_num_samples=num_samples,
                        n_values=capped_n_values,
                        sample_batch_size=sample_batch_size,
                        score_batch_size=score_batch_size,
                        bos_token_id=bos_token_id,
                        seed=seed,
                        sample_cache_path=sample_cache_path,
                        autocast_dtype=autocast_dtype,
                        save_every_batches=save_every_batches,
                        token_log_every=token_log_every,
                        progress_desc=f"Sampling+scoring hf model={model_name}",
                    )
                )
            else:
                raise
    finally:
        if model is not None:
            del model
        if torch_module is not None and device is not None and device.type == "cuda":
            torch_module.cuda.empty_cache()

    if sample_logps is None or sample_logps.shape[0] < int(num_samples):
        raise RuntimeError("Missing sampled log-prob cache for HF sampled estimator")

    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)
    mi_values = _compute_bipartite_mi_from_sampled_q(
        sample_logps=sample_logps,
        n_values=capped_n_values,
        log_q_y_means_by_n=log_q_y_means_by_n,
    )
    return mi_values, max_positions


def _plot_bipartite_mi(
    values: dict[str, dict[int, dict[int, float]]],
    estimators: list[str],
    out_path: str,
    title: str,
    extra_series: list[dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))

    def _fit_power_law(
        ns: np.ndarray,
        ys: np.ndarray,
        nmax: int = FIT_NMAX,
    ) -> tuple[float, float] | None:
        fit_mask = (ns < float(nmax)) & np.isfinite(ys) & (ys > 0.0)
        if int(np.sum(fit_mask)) < 2:
            return None
        x = np.log(ns[fit_mask])
        y = np.log(ys[fit_mask])
        if x.size < 2:
            return None
        power, log_const = np.polyfit(x, y, 1)
        const = float(np.exp(log_const))
        if not np.isfinite(const) or not np.isfinite(power):
            return None
        return const, float(power)

    hidden_dims = sorted(
        {
            hidden_dim
            for estimator_values in values.values()
            for hidden_dim in estimator_values
        }
    )
    if not hidden_dims:
        raise RuntimeError("No MI values available to plot")

    cmap = plt.cm.viridis
    if len(hidden_dims) == 1:
        norm = plt.Normalize(vmin=hidden_dims[0] - 1, vmax=hidden_dims[0] + 1)
    else:
        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))

    line_styles = {
        "logged": "-",
        "sampled": "--",
    }
    markers = {
        "logged": "o",
        "sampled": "s",
    }
    show_marker = len(estimators) > 1
    fit_handles_by_hidden_dim: dict[int, Line2D] = {}
    hf_fit_handle: Line2D | None = None

    for estimator in estimators:
        estimator_values = values.get(estimator, {})
        for hidden_dim in hidden_dims:
            series = estimator_values.get(hidden_dim)
            if not series:
                continue
            ns = np.array(sorted(series.keys()), dtype=float)
            s_ab = np.array([series[int(n)] for n in ns], dtype=float)
            ax.plot(
                ns,
                s_ab,
                color=cmap(norm(hidden_dim)),
                linestyle=line_styles.get(estimator, "-"),
                marker=markers.get(estimator, "o") if show_marker else None,
                markersize=3 if show_marker else None,
            )
            fit = _fit_power_law(ns, s_ab)
            if fit is not None:
                const, power = fit
                fit_ns_max = float(np.max(ns[ns < FIT_NMAX]))
                fit_ns = np.logspace(
                    np.log10(float(np.min(ns))),
                    np.log10(fit_ns_max),
                    num=200,
                )
                fit_curve = const * np.power(fit_ns, power)
                ax.plot(
                    fit_ns,
                    fit_curve,
                    color=cmap(norm(hidden_dim)),
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
                print(
                    f"fit estimator={estimator} hidden_dim={hidden_dim}: "
                    f"MI={const:.4g}*N^{power:.4f} (N<128)"
                )
                if estimator == "sampled" or hidden_dim not in fit_handles_by_hidden_dim:
                    fit_handles_by_hidden_dim[hidden_dim] = Line2D(
                        [0],
                        [0],
                        color=cmap(norm(hidden_dim)),
                        linestyle=":",
                        linewidth=1.5,
                        label=(
                            rf"$d_h={hidden_dim}$: "
                            rf"$I(A:B)={const:.3g}\cdot N^{{{power:.3f}}}$"
                        ),
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=6)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$I(A:B)$")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("hidden_dim")

    if extra_series is not None:
        for series_info in extra_series:
            series = series_info.get("series")
            if not series:
                continue
            ns = np.array(sorted(series.keys()), dtype=float)
            mi = np.array([series[int(n)] for n in ns], dtype=float)
            ax.plot(
                ns,
                mi,
                color=series_info.get("color", "black"),
                linestyle=series_info.get("linestyle", "-."),
                marker=series_info.get("marker", "^"),
                markersize=4,
                linewidth=1.5,
                label=series_info.get("label", "hf_model"),
            )
            fit = _fit_power_law(ns, mi)
            if fit is not None:
                const, power = fit
                fit_ns_max = float(np.max(ns[ns < FIT_NMAX]))
                fit_ns = np.logspace(
                    np.log10(float(np.min(ns))),
                    np.log10(fit_ns_max),
                    num=200,
                )
                fit_curve = const * np.power(fit_ns, power)
                ax.plot(
                    fit_ns,
                    fit_curve,
                    color=series_info.get("color", "black"),
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
                print(
                    f"fit series={series_info.get('label', 'hf_model')}: "
                    f"MI={const:.4g}*N^{power:.4f} (N<128)"
                )
                hf_fit_handle = Line2D(
                    [0],
                    [0],
                    color=series_info.get("color", "black"),
                    linestyle=":",
                    linewidth=1.5,
                    label=rf"GPT-2-1.5B: $I(A:B)={const:.3g}\cdot N^{{{power:.3f}}}$",
                )
            elif hf_fit_handle is None:
                hf_fit_handle = Line2D(
                    [0],
                    [0],
                    color=series_info.get("color", "black"),
                    linestyle=":",
                    linewidth=1.5,
                    label="GPT-2-1.5B: fit failed",
                )

    legend_handles = [
        fit_handles_by_hidden_dim[hidden_dim]
        for hidden_dim in sorted(fit_handles_by_hidden_dim)
    ]
    if hf_fit_handle is not None:
        legend_handles.append(hf_fit_handle)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=1,
            borderaxespad=0.0,
            labelspacing=0.25,
            handletextpad=0.4,
            frameon=False,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    core = parser.add_argument_group("core")
    core.add_argument("--group", type=str, required=True)
    core.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    core.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Maximum N to include in log-spaced N grid",
    )
    core.add_argument(
        "--num-n-values",
        type=int,
        default=DEFAULT_NUM_N_VALUES,
        help="Number of log-spaced N values",
    )
    core.add_argument(
        "--estimator",
        type=str,
        choices=["logged", "sampled"],
        nargs="+",
        default=["logged"],
        help="One or more estimators: logged sampled",
    )
    core.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )

    sampled = parser.add_argument_group("sampled estimator")
    sampled.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of sampled sequences for sampled estimator",
    )
    sampled.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling and scoring in sampled estimator",
    )
    sampled.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
        help="Directory for cached downloaded checkpoints",
    )
    sampled.add_argument(
        "--force-resample",
        action="store_true",
        help="Force regeneration of sampled caches instead of cache-only mode",
    )

    hf = parser.add_argument_group("hf sampled overlay")
    hf.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Optional HF causal LM to evaluate with sampled MI",
    )
    hf.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional HF model revision",
    )
    hf.add_argument(
        "--hf-batch-size",
        type=int,
        default=None,
        help=(
            "Legacy shared HF batch size for sampling+scoring "
            "(defaults to --batch-size)"
        ),
    )
    hf.add_argument(
        "--hf-sample-batch-size",
        type=int,
        default=None,
        help="HF sampling batch size (defaults to --hf-batch-size or --batch-size)",
    )
    hf.add_argument(
        "--hf-score-batch-size",
        type=int,
        default=None,
        help="HF q(y) scoring batch size (defaults to --hf-batch-size or --batch-size)",
    )
    hf.add_argument(
        "--hf-save-every-batches",
        type=int,
        default=DEFAULT_HF_SAVE_EVERY_BATCHES,
        help="Persist partial HF sample cache every this many batches",
    )
    hf.add_argument(
        "--hf-token-log-every",
        type=int,
        default=1,
        help="Update HF token-step progress every this many decoded steps",
    )
    hf.add_argument(
        "--hf-compile",
        type=str,
        choices=["none", "score", "sample", "both"],
        default="none",
        help="Optional torch.compile target for HF path",
    )
    hf.add_argument(
        "--hf-compile-mode",
        type=str,
        default="default",
        help="torch.compile mode for HF path",
    )
    hf.add_argument(
        "--hf-attn-implementation",
        type=str,
        default="sdpa",
        help="HF attention backend (e.g. sdpa, eager, flash_attention_2, none)",
    )
    hf.add_argument(
        "--hf-seed",
        type=int,
        default=0,
        help="Sampling seed for HF model",
    )
    hf.add_argument(
        "--hf-label",
        type=str,
        default=None,
        help="Legend label for HF model curve",
    )
    hf.add_argument(
        "--hf-num-samples",
        type=int,
        default=None,
        help="HF-only number of sampled sequences (defaults to --num-samples)",
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_n < DEFAULT_MIN_N:
        raise RuntimeError(f"--max-n must be >= {DEFAULT_MIN_N}")
    if args.num_n_values < 1:
        raise RuntimeError("--num-n-values must be >= 1")
    if args.hf_batch_size is not None and args.hf_batch_size < 1:
        raise RuntimeError("--hf-batch-size must be >= 1")
    if args.hf_sample_batch_size is not None and args.hf_sample_batch_size < 1:
        raise RuntimeError("--hf-sample-batch-size must be >= 1")
    if args.hf_score_batch_size is not None and args.hf_score_batch_size < 1:
        raise RuntimeError("--hf-score-batch-size must be >= 1")
    if args.hf_save_every_batches < 1:
        raise RuntimeError("--hf-save-every-batches must be >= 1")
    if args.hf_token_log_every < 1:
        raise RuntimeError("--hf-token-log-every must be >= 1")
    if args.hf_num_samples is not None and args.hf_num_samples < 1:
        raise RuntimeError("--hf-num-samples must be >= 1")


def _dedupe_estimators(estimator_args: list[str]) -> list[str]:
    estimators: list[str] = []
    for estimator in estimator_args:
        if estimator not in estimators:
            estimators.append(estimator)
    return estimators


def _filter_runs_by_hidden_dim(
    runs: list[wandb.apis.public.Run],
    hidden_dim_filter: list[int] | None,
    group: str,
) -> list[wandb.apis.public.Run]:
    if hidden_dim_filter is None:
        return runs
    hidden_dims = set(hidden_dim_filter)
    filtered_runs = [
        run
        for run in runs
        if int((run.config or {}).get("hidden_dim", -1)) in hidden_dims
    ]
    if not filtered_runs:
        raise RuntimeError(
            f"No finished runs found for group='{group}' "
            f"hidden_dim in {sorted(hidden_dims)}"
        )
    return filtered_runs


def _select_cached_n_values(
    requested_n_values: list[int],
    log_q_y_means_by_n: dict[int, float],
    sample_logps: np.ndarray,
) -> list[int]:
    if not requested_n_values:
        return []
    max_requested_n = int(max(requested_n_values))
    max_allowed_n = min(max_requested_n, int(sample_logps.shape[1]))
    cached_n_values = sorted(
        int(n)
        for n in log_q_y_means_by_n
        if DEFAULT_MIN_N <= int(n) <= max_allowed_n and int(n) % 2 == 0
    )
    if cached_n_values:
        return cached_n_values
    return sorted(
        n
        for n in requested_n_values
        if n in log_q_y_means_by_n and n <= int(sample_logps.shape[1])
    )


def _compute_logged_mi_for_run(
    run: wandb.apis.public.Run,
    hidden_dim: int,
    n_values: list[int],
) -> dict[int, float]:
    conditional_entropy = _extract_conditional_entropy(run)
    if not conditional_entropy:
        raise RuntimeError(
            "No conditional entropy metrics found for "
            f"run '{run.name}' (hidden_dim={hidden_dim})"
        )
    mi_values = _compute_bipartite_mi_from_conditional_entropy(
        conditional_entropy,
        n_values,
    )
    if not mi_values:
        raise RuntimeError(
            "Insufficient conditional entropy metrics to compute bipartite MI for "
            f"run '{run.name}' (hidden_dim={hidden_dim})"
        )
    return mi_values


def _compute_lstm_sampled_mi_for_run(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    hidden_dim: int,
    n_values: list[int],
    *,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    sample_key = _sample_cache_key(
        seq_len=n_values[-1],
        num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
    )
    sample_cache_path, log_q_y_cache_path = _sample_cache_paths(
        cache_dir,
        run.id,
        sample_key,
    )

    if not force_resample:
        reusable_cache = _find_reusable_complete_cache(
            cache_dir=cache_dir,
            run_id=run.id,
            seq_len=n_values[-1],
            target_num_samples=num_samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            n_values=n_values,
        )
        if reusable_cache is None:
            raise RuntimeError(
                "No complete cached sampled MI found for "
                f"hidden_dim={hidden_dim}. Re-run with --force-resample to regenerate."
            )
        (
            reusable_sample_logps,
            reusable_log_q_y_means_by_n,
            reusable_num_samples,
            reusable_sample_cache_path,
            _,
        ) = reusable_cache
        available_n_values = _select_cached_n_values(
            n_values,
            reusable_log_q_y_means_by_n,
            reusable_sample_logps,
        )
        if not available_n_values:
            raise RuntimeError(
                "Cached sampled artifacts have no usable N values for "
                f"hidden_dim={hidden_dim}. Re-run with --force-resample to regenerate."
            )
        if len(available_n_values) < len(n_values):
            print(
                "Using cached sampled MI subset for hidden_dim="
                f"{hidden_dim}: {len(available_n_values)}/{len(n_values)} "
                "N values available"
            )
        mi_values = _compute_bipartite_mi_from_sampled_q(
            sample_logps=reusable_sample_logps,
            n_values=available_n_values,
            log_q_y_means_by_n=reusable_log_q_y_means_by_n,
        )
        print(
            "Using cached sampled MI for hidden_dim="
            f"{hidden_dim} from {os.path.basename(reusable_sample_cache_path)} "
            f"(num_samples={reusable_num_samples}, requested={num_samples})"
        )
        return mi_values

    print(
        "Force resample enabled for hidden_dim="
        f"{hidden_dim}; regenerating sampled cache"
    )
    if os.path.exists(sample_cache_path):
        os.remove(sample_cache_path)
    if os.path.exists(log_q_y_cache_path):
        os.remove(log_q_y_cache_path)

    import jax

    from models.lstm import LSTMLanguageModel
    from training.trainer import create_train_state

    ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
    model = LSTMLanguageModel(
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]),
    )
    rng = jax.random.PRNGKey(0)
    state_cfg = SimpleNamespace(
        batch_size=int(cfg.get("batch_size", 1)),
        seq_len=int(cfg.get("seq_len", n_values[-1])),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            "Checkpoint/run mismatch: " f"ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    sample_params = _normalize_params_for_step(
        state.params,
        int(cfg["num_layers"]),
    )
    samples, sample_logps = _sample_sequences(
        model=model,
        params=sample_params,
        seq_len=n_values[-1],
        num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        rng=rng,
        progress_desc=f"Sampling d_h={hidden_dim}",
    )
    _save_sample_cache(sample_cache_path, samples, sample_logps)

    log_q_y_means_by_n = _compute_log_q_y_means(
        samples=samples,
        apply_fn=state.apply_fn,
        params=sample_params,
        n_values=n_values,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc_prefix=f"Scoring y d_h={hidden_dim}",
    )
    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)

    return _compute_bipartite_mi_from_sampled_q(
        sample_logps=sample_logps,
        n_values=n_values,
        log_q_y_means_by_n=log_q_y_means_by_n,
    )


def _resolve_hf_series_options(
    args: argparse.Namespace,
) -> tuple[int, int, int, str | None]:
    hf_num_samples = (
        args.num_samples if args.hf_num_samples is None else args.hf_num_samples
    )
    hf_default_batch_size = (
        args.batch_size if args.hf_batch_size is None else args.hf_batch_size
    )
    hf_sample_batch_size = (
        hf_default_batch_size
        if args.hf_sample_batch_size is None
        else args.hf_sample_batch_size
    )
    hf_score_batch_size = (
        hf_default_batch_size
        if args.hf_score_batch_size is None
        else args.hf_score_batch_size
    )
    hf_attn_implementation = (
        None
        if str(args.hf_attn_implementation).lower() == "none"
        else args.hf_attn_implementation
    )
    return (
        hf_num_samples,
        hf_sample_batch_size,
        hf_score_batch_size,
        hf_attn_implementation,
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    n_values = [n for n in DEFAULT_N_VALUES if n <= int(args.max_n)]
    if not n_values:
        raise RuntimeError("No valid N values to evaluate")

    estimators = _dedupe_estimators(args.estimator)

    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, args.group)

    all_values: dict[str, dict[int, dict[int, float]]] = {
        estimator: {} for estimator in estimators
    }
    for run in tqdm(runs, desc="Runs", unit="run"):
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])

        if "logged" in estimators:
            all_values["logged"][hidden_dim] = _compute_logged_mi_for_run(
                run,
                hidden_dim,
                n_values,
            )

        if "sampled" in estimators:
            all_values["sampled"][hidden_dim] = _compute_lstm_sampled_mi_for_run(
                run,
                api,
                hidden_dim,
                n_values,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
                force_resample=args.force_resample,
            )

    extra_series: list[dict[str, Any]] = []
    if args.hf_model is not None:
        (
            hf_num_samples,
            hf_sample_batch_size,
            hf_score_batch_size,
            hf_attn_implementation,
        ) = _resolve_hf_series_options(args)

        hf_mi_values, hf_max_positions = _compute_hf_sampled_mi_series(
            model_name=args.hf_model,
            revision=args.hf_revision,
            n_values=n_values,
            num_samples=hf_num_samples,
            sample_batch_size=hf_sample_batch_size,
            score_batch_size=hf_score_batch_size,
            cache_dir=args.cache_dir,
            seed=args.hf_seed,
            save_every_batches=args.hf_save_every_batches,
            token_log_every=args.hf_token_log_every,
            compile_target=args.hf_compile,
            compile_mode=args.hf_compile_mode,
            attn_implementation=hf_attn_implementation,
            force_resample=args.force_resample,
        )
        if hf_max_positions < max(n_values):
            print(
                f"HF model context limit={hf_max_positions}; plotting only N <= "
                f"{hf_max_positions} for {args.hf_model}"
            )
        extra_series.append(
            {
                "label": (
                    args.hf_label
                    if args.hf_label is not None
                    else f"{args.hf_model} (sampled)"
                ),
                "series": hf_mi_values,
                "color": "black",
                "linestyle": "-.",
                "marker": "^",
            }
        )

    out_path = (
        args.output
        if args.output is not None
        else f"results/bipartite_mi_{args.group}.png"
    )
    _plot_bipartite_mi(
        all_values,
        estimators,
        out_path,
        title=f"Bipartite MI (group={args.group})",
        extra_series=extra_series,
    )


if __name__ == "__main__":
    main()
