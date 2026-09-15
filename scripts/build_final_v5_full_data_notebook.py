"""Build the clean, final-fit arrival-v5 notebook.

This generator only writes notebook JSON. It never prepares data or trains a model.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "rain_arrival_v5_full_data.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Final full-archive training — arrival v5

This notebook trains the retained **arrival v5 architecture** on every eligible radar/weather window before a frozen cutoff.

It is intentionally a **final fit**, not another architecture experiment:

- six 5-minute history frames, seven existing channels;
- compact CNN encoder → one 64-channel ConvGRU → global pooling → 13 categorical arrival classes;
- the corrected four-group dynamic sampler;
- no hard-negative enrichment;
- strict deterministic execution;
- fixed update budget, because using all current dates leaves no honest validation set for early stopping;
- resumable checkpoints containing optimizer and random-number-generator state;
- no calibration, notification replay, deployment, or production-file changes.

`LAST_TRAINING_DATE = "2026-09-14"` includes all eligible archived data through 14 September. The notebook derives the exclusive boundary of 15 September automatically; anything collected from then onward is the prospective test period. Because all current data trains the weights, this run has **no independent present-day performance estimate**, and an older calibrator must not be reused with it.

## Run instructions

1. Shut down other GPU notebook kernels.
2. Restart this notebook's kernel so the deterministic environment is set before CUDA initializes.
3. Review the configuration cell.
4. Run All. Preparation may be slow the first time; training resumes from `latest.pt` after interruption.
5. A completed run writes `final.pt`, `result.json`, and `verification.json` under `models/arrival/final_v5_full_through_20260914/seed67/`.
"""
    ),
    code(
        """# This must be the first code cell in a freshly restarted kernel.
import os
import sys

if "torch" in sys.modules:
    raise RuntimeError("PyTorch is already imported. Restart the kernel, then Run All for strict determinism.")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
"""
    ),
    code(
        """from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import random
import sys
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path.cwd().resolve()
if not (ROOT / "src" / "arrival").is_dir():
    if (ROOT.parent / "src" / "arrival").is_dir():
        ROOT = ROOT.parent
    else:
        raise FileNotFoundError("Run this notebook from the repository root or src directory")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arrival.data import (
    ArrivalDataset,
    DataConfig,
    FrameStore,
    LabelStore,
    fingerprint,
    fit_normalization,
    shifted,
)
from arrival.models import ArrivalNet
from arrival.training import seed_everything
from data_processing.data_loading import RADAR_CLEANING_VERSION
from data_processing.radar_codec import SOURCE

print("Repository:", ROOT)
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
"""
    ),
    markdown(
        """## Frozen configuration

Changing the cutoff, seed, architecture, labels, channels, sampler, or update budget creates a different model. Use a new output directory rather than overwriting this run.
"""
    ),
    code(
        """LAST_TRAINING_DATE = "2026-09-14"  # included
DATA_END_EXCLUSIVE = (datetime.fromisoformat(LAST_TRAINING_DATE) + timedelta(days=1)).strftime("%Y-%m-%d")
SEED = 67
UPDATE_BUDGET = 40_000
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT_EVERY = 1_000
SAMPLES_PER_ANCHOR = 32
NORMALIZATION_FRAMES = 512
RUN_PREPARATION = True
RUN_TRAINING = True

DATA_CONFIG = DataConfig(
    crop=64,
    max_history=12,
    future=12,
    patch=5,
    rain_threshold=0.01,
    coverage=0.20,
    stride=1,
    development_end=DATA_END_EXCLUSIVE,
    test_start=None,
    seed=SEED,
)
MODEL_CONFIG = dict(
    history=6,
    channels=7,
    kind="gru",
    output="categorical",
    dropout=0.15,
    hidden_dim=64,
    pooling="global",
    recurrent_layers=1,
    encoder_extra_layers=0,
)
CHANNEL_INDICES = tuple(range(7))
RUN_ROOT = ROOT / "models" / "arrival" / "final_v5_full_through_20260914" / f"seed{SEED}"
DATA_DIR = ROOT / "models" / "arrival" / "final_v5_full_through_20260914" / "data"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

assert MODEL_CONFIG == dict(history=6, channels=7, kind="gru", output="categorical", dropout=0.15,
                            hidden_dim=64, pooling="global", recurrent_layers=1, encoder_extra_layers=0)
assert DATA_CONFIG.max_history == 12 and DATA_CONFIG.future == 12
assert UPDATE_BUDGET > 0 and BATCH_SIZE > 0 and CHECKPOINT_EVERY > 0
print("Output:", RUN_ROOT)
print("Last included training date:", LAST_TRAINING_DATE)
print("Derived data end (exclusive):", DATA_END_EXCLUSIVE)
"""
    ),
    markdown(
        """## Build one full-data manifest

Every admitted anchor has all 12 common-history frames, all 12 future label frames, and causally valid weather for every historical input. Missing observations create gaps; they are never interpreted as dry weather. No date before the cutoff is reserved for model selection or calibration.
"""
    ),
    code(
        """def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_inventory(store: FrameStore, end_exclusive: str) -> tuple[list[str], str]:
    end_key = datetime.fromisoformat(end_exclusive).strftime("%Y%m%d%H%M")
    rows = []
    for key in store.keys:
        if key >= end_key:
            continue
        radar = store.radar_dir / f"{key}.png"
        weather = store.env_dir / f"weather_{key}.csv"
        rows.append((key, radar.stat().st_size, radar.stat().st_mtime_ns,
                     weather.stat().st_size if weather.exists() else None,
                     weather.stat().st_mtime_ns if weather.exists() else None))
    return [row[0] for row in rows], fingerprint(rows)


def complete_full_anchors(store: FrameStore, config: DataConfig, end_exclusive: str) -> list[str]:
    end_key = datetime.fromisoformat(end_exclusive).strftime("%Y%m%d%H%M")
    anchors = []
    for key in store.keys:
        if key >= end_key:
            continue
        history = [shifted(key, -5 * j) for j in range(config.max_history - 1, -1, -1)]
        future = [shifted(key, 5 * j) for j in range(1, config.future + 1)]
        window = history + future
        if not all(item in store.key_set and item < end_key for item in window):
            continue
        if not all(store.environment_valid(item) for item in history):
            continue
        anchors.append(key)
    if not anchors:
        raise ValueError("No complete eligible anchors before the frozen cutoff")
    return anchors


store = FrameStore(ROOT)
raw_keys, input_inventory_hash = source_inventory(store, DATA_END_EXCLUSIVE)
metadata_path = DATA_DIR / "prepared_full.json"

if RUN_PREPARATION:
    labels = LabelStore(store, DATA_CONFIG)
    candidate_anchors = complete_full_anchors(store, DATA_CONFIG, DATA_END_EXCLUSIVE)
    if metadata_path.exists():
        prepared = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "data_config": asdict(DATA_CONFIG),
            "data_end_exclusive": DATA_END_EXCLUSIVE,
            "input_inventory_hash": input_inventory_hash,
        }
        for key, value in expected.items():
            if prepared.get(key) != value:
                raise ValueError(f"Prepared full-data metadata mismatch for {key}; use a new versioned directory")
        anchors = prepared["anchors"]
        norm = prepared["norm"]
        print("Reused prepared metadata:", metadata_path)
    else:
        print(f"Inspecting labels for {len(candidate_anchors):,} temporally complete anchors...")
        anchors, class_counts = labels.inspect(candidate_anchors)
        norm = fit_normalization(store, anchors, NORMALIZATION_FRAMES, DATA_CONFIG.max_history)
        prepared = {
            "schema": "arrival_full_fit_data_v1",
            "data_config": asdict(DATA_CONFIG),
            "data_end_exclusive": DATA_END_EXCLUSIVE,
            "raw_radar_keys_before_cutoff": len(raw_keys),
            "candidate_anchors": len(candidate_anchors),
            "anchors": anchors,
            "class_counts": class_counts.tolist(),
            "norm": norm,
            "anchor_hash": fingerprint(anchors),
            "input_inventory_hash": input_inventory_hash,
            "decoder": SOURCE,
            "cleaning": RADAR_CLEANING_VERSION,
            "caveat": "All pre-cutoff dates fit model weights; no current independent validation/calibration partition remains.",
        }
        atomic_json(metadata_path, prepared)
        print("Wrote:", metadata_path)
else:
    raise RuntimeError("Preparation is required for this clean final-fit notebook")

assert anchors == sorted(anchors)
assert max(anchors) < datetime.fromisoformat(DATA_END_EXCLUSIVE).strftime("%Y%m%d%H%M")
assert prepared["anchor_hash"] == fingerprint(anchors)
print(f"Eligible full-data anchors: {len(anchors):,}")
print("First / last anchor:", anchors[0], anchors[-1])
print("Natural candidate-location class counts:", prepared["class_counts"])
print("Input inventory hash:", input_inventory_hash)
"""
    ),
    markdown(
        """## Dataset and deterministic model initialization

The training sampler remains balanced across 0–15, 15–30, 30–60, and no-arrival groups. Hard-negative enrichment is explicitly disabled. Sequential index traversal makes checkpoint resume reproducible; dynamic samples still vary deterministically by epoch and index.
"""
    ),
    code(
        """seed_everything(SEED, deterministic=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is required for the intended full training run")
print("Device:", device, torch.cuda.get_device_name(0))

train_data = ArrivalDataset(
    store=store,
    labels=labels,
    anchors=anchors,
    norm=norm,
    history=MODEL_CONFIG["history"],
    channels=CHANNEL_INDICES,
    per_anchor=SAMPLES_PER_ANCHOR,
    seed=SEED,
    hard_negative_fraction=0.0,
)
train_data.build_group_index()
assert train_data.hard_negative_fraction == 0.0

model = ArrivalNet(**MODEL_CONFIG).to(device)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert parameter_count == 174_253, parameter_count
print("Training samples per sampler epoch:", f"{len(train_data):,}")
print("Parameters:", f"{parameter_count:,}")
"""
    ),
    markdown(
        """## Fixed-budget, interruption-safe final training

There is deliberately no `best.pt`: without held-out development data, calling any checkpoint “best” would be misleading. `latest.pt` is an exact resumable training state at a checkpoint boundary, and `final.pt` is the fixed-budget artifact.
"""
    ),
    code(
        """def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(model, optimizer, step, epoch, next_index, elapsed_seconds, losses):
    return {
        "schema": "arrival_v5_full_fit_checkpoint_v1",
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": MODEL_CONFIG,
        "data_config": asdict(DATA_CONFIG),
        "norm": norm,
        "channels": CHANNEL_INDICES,
        "seed": SEED,
        "step": step,
        "epoch": epoch,
        "next_index": next_index,
        "budget": UPDATE_BUDGET,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "anchor_hash": prepared["anchor_hash"],
        "input_inventory_hash": input_inventory_hash,
        "sampler": "global_groups_v2",
        "hard_negative_fraction": 0.0,
        "selection": "fixed_budget_no_validation",
        "elapsed_seconds": elapsed_seconds,
        "recent_losses": losses[-100:],
        "rng_state": rng_state(),
        "reproducibility": {
            "deterministic": True,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


latest_path = RUN_ROOT / "latest.pt"
final_path = RUN_ROOT / "final.pt"
result_path = RUN_ROOT / "result.json"
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
step = 0
epoch = 0
next_index = 0
elapsed_before = 0.0
losses = []

if final_path.exists() or result_path.exists():
    raise FileExistsError(f"Completed artifact already exists in {RUN_ROOT}; do not overwrite a final run")
if latest_path.exists():
    resume = torch.load(latest_path, map_location=device, weights_only=False)
    required_identity = {
        "model_config": MODEL_CONFIG,
        "data_config": asdict(DATA_CONFIG),
        "seed": SEED,
        "budget": UPDATE_BUDGET,
        "batch_size": BATCH_SIZE,
        "anchor_hash": prepared["anchor_hash"],
        "input_inventory_hash": input_inventory_hash,
        "hard_negative_fraction": 0.0,
    }
    for key, value in required_identity.items():
        if resume.get(key) != value:
            raise ValueError(f"Resume checkpoint mismatch for {key}")
    model.load_state_dict(resume["state_dict"])
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    step, epoch, next_index = resume["step"], resume["epoch"], resume["next_index"]
    elapsed_before = resume.get("elapsed_seconds", 0.0)
    losses = list(resume.get("recent_losses", []))
    restore_rng(resume["rng_state"])
    print(f"Resuming exact checkpoint boundary at step {step:,}, epoch {epoch}, sample index {next_index:,}")

if RUN_TRAINING:
    started = time.monotonic()
    while step < UPDATE_BUDGET:
        train_data.set_epoch(epoch)
        train_data.set_step(step)
        if next_index >= len(train_data):
            epoch += 1
            next_index = 0
            continue
        indices = range(next_index, min(next_index + BATCH_SIZE, len(train_data)))
        examples = [train_data[index] for index in indices]
        x = torch.stack([example[0] for example in examples]).to(device)
        y = torch.as_tensor([example[1] for example in examples], dtype=torch.long, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.nll_loss(model.log_probs(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        next_index += len(examples)
        train_data.set_step(step)
        losses.append(float(loss.detach().cpu()))

        if step % 250 == 0 or step == UPDATE_BUDGET:
            mean_recent = float(np.mean(losses[-250:]))
            print({"step": step, "epoch": epoch, "mean_recent_training_loss": mean_recent}, flush=True)
        if step % CHECKPOINT_EVERY == 0 or step == UPDATE_BUDGET:
            elapsed = elapsed_before + time.monotonic() - started
            payload = checkpoint_payload(model, optimizer, step, epoch, next_index, elapsed, losses)
            atomic_torch_save(latest_path, payload)

    final_state = torch.load(latest_path, map_location="cpu", weights_only=False)
    if final_state["step"] != UPDATE_BUDGET:
        raise RuntimeError("Latest checkpoint did not reach the fixed budget")
    atomic_torch_save(final_path, final_state)
    result = {
        "schema": "arrival_v5_full_fit_result_v1",
        "status": "completed",
        "checkpoint": final_path.name,
        "model_config": MODEL_CONFIG,
        "parameters": parameter_count,
        "seed": SEED,
        "actual_updates": final_state["step"],
        "elapsed_seconds": final_state["elapsed_seconds"],
        "data_end_exclusive": DATA_END_EXCLUSIVE,
        "eligible_anchors": len(anchors),
        "anchor_hash": prepared["anchor_hash"],
        "input_inventory_hash": input_inventory_hash,
        "sampler": "global_groups_v2",
        "hard_negative_fraction": 0.0,
        "calibrated": False,
        "independent_validation": False,
        "deployment_ready": False,
        "next_valid_evaluation": f"Data collected at or after {DATA_END_EXCLUSIVE}, untouched by model development.",
    }
    atomic_json(result_path, result)
    print("Completed fixed-budget final fit:", final_path)
else:
    print("RUN_TRAINING is False; preparation and model construction completed without training")
"""
    ),
    markdown(
        """## Post-run integrity verification

This verifies artifact identity and reloadability. It does **not** estimate predictive skill.
"""
    ),
    code(
        """if result_path.exists() and final_path.exists():
    result = json.loads(result_path.read_text(encoding="utf-8"))
    state = torch.load(final_path, map_location="cpu", weights_only=False)
    reloaded = ArrivalNet(**state["model_config"])
    reloaded.load_state_dict(state["state_dict"], strict=True)
    final_sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()
    checks = {
        "checkpoint_loads_strictly": True,
        "architecture_is_v5": state["model_config"] == MODEL_CONFIG,
        "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "reached_fixed_budget": state["step"] == UPDATE_BUDGET,
        "anchor_hash_matches": state["anchor_hash"] == prepared["anchor_hash"],
        "input_inventory_hash_matches": state["input_inventory_hash"] == input_inventory_hash,
        "strict_determinism_recorded": state["reproducibility"]["deterministic"] is True,
        "no_hard_negative_sampling": state["hard_negative_fraction"] == 0.0,
        "no_calibrator_attached": result["calibrated"] is False,
        "checkpoint_sha256": final_sha256,
    }
    if not all(value is True for key, value in checks.items() if key not in ("parameter_count", "checkpoint_sha256")):
        raise AssertionError(checks)
    if checks["parameter_count"] != 174_253:
        raise AssertionError(checks)
    atomic_json(RUN_ROOT / "verification.json", checks)
    print(json.dumps(checks, indent=2))
else:
    print("No completed final artifact yet; run the training cell first")
"""
    ),
    markdown(
        """## What happens next

Do not calibrate this checkpoint using any date that trained its weights, and never reuse the old v5 calibrator. Keep collecting radar and causal weather from 15 September onward. Once a substantial prospective period has accumulated, evaluate this frozen checkpoint once against the retained historical v5 reference. Only a model calibrated on separate representative data can provide user-facing probabilities or notification thresholds.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
