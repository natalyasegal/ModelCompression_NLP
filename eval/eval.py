# eval.py
# Evaluates all saved model versions under outputs_dir on CPU, produces:
# - comparison table (printed + optional CSV)
# - per-model confusion_matrix.png (counts + row-%)
#
# Works with:
# 1) Standard HuggingFace saved model dirs (config.json present)
# 2) Dynamic INT8 quant dirs created via meta.json (with base_model_dir)

import os
import time
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)

# ----------------------------
# Core helpers
# ----------------------------
def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


@torch.no_grad()
def _predict_labels(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    model.eval().to(device)
    preds_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds_all.append(logits.argmax(dim=-1).detach().cpu().numpy())
    return np.concatenate(preds_all, axis=0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> Dict[str, float]:
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    for c in range(num_labels):
        mask = (y_true == c)
        out[f"acc_c{c}"] = float((y_pred[mask] == y_true[mask]).mean()) if mask.sum() else np.nan

        # per-class F1 as one-vs-rest
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        out[f"f1_c{c}"] = float(f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0))
    return out


@torch.no_grad()
def _benchmark_ms_per_sample(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
    warmup: int = 2,
) -> float:
    model.eval().to(device)

    # warmup
    for _ in range(warmup):
        enc = tokenizer(
            texts[:batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc).logits

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc).logits
        n += len(batch)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(n, 1)


def _load_model_from_dir(model_dir: str, num_labels: int):
    """
    Loads:
      - Standard HF saved model dir (config.json present)
      - INT8 dynamic quant dir (meta.json present with base_model_dir)
    """
    meta_path = os.path.join(model_dir, "meta.json")
    config_path = os.path.join(model_dir, "config.json")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # INT8 dynamic reconstruction (CPU-only)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        base_dir = meta.get("base_model_dir")
        if not base_dir or not os.path.exists(base_dir):
            raise FileNotFoundError(f"{meta_path} exists but base_model_dir is missing/invalid: {base_dir}")

        base_model = AutoModelForSequenceClassification.from_pretrained(base_dir, num_labels=num_labels)
        base_model.eval().to("cpu")
        qmodel = torch.quantization.quantize_dynamic(base_model, {nn.Linear}, dtype=torch.qint8)
        return tokenizer, qmodel, "cpu"

    # Regular HF model
    if os.path.exists(config_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        return tokenizer, model, None

    raise FileNotFoundError(f"Not a recognizable model directory (no config.json / meta.json): {model_dir}")


# ----------------------------
# Confusion matrix plotting
# ----------------------------
def plot_confusion_matrix_counts_and_percents(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_png: str,
    normalize: str = "true",  # row-normalized percents
    title: str = "Confusion Matrix (counts and %)",
    show: bool = False,
) -> None:
    """
    Saves a confusion matrix plot with counts + percents per cell.
    normalize="true" => rows sum to 1 (percent of each true class).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = len(class_names)
    labels = list(range(n))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize) if normalize else None

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid(False)
    #im = ax.imshow(cm, interpolation="nearest",  cmap="Blues")
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick = np.arange(n)
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    max_val = cm.max() if cm.size else 0
    thresh = max_val / 2.0 if max_val else 0.0

    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            if cm_norm is not None:
                pct = float(cm_norm[i, j]) * 100.0
                txt = f"{count}\n{pct:.1f}%"
            else:
                txt = f"{count}"

            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if cm_norm[i, j] >= 0.5 else "black",
                fontsize=10,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------
# Main evaluation function
# ----------------------------
def evaluate_all_versions_from_outputs(
    outputs_dir: str,
    df_v: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    num_labels: int = 6,
    device: str = "cpu",              # you want CPU evaluation
    max_length: int = 128,
    batch_size: int = 32,
    bench_samples: int = 512,
    save_csv_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    save_confusion_matrices: bool = True,
    show_confusion_matrices: bool = False,
) -> pd.DataFrame:
    """
    Reads every subdirectory in outputs_dir as a model version.
    Evaluates on df_v (CPU by default).
    Saves confusion_matrix.png in each model directory (optional).
    Returns a DataFrame with metrics, model_size_mb, inference time, and model_name from directory name.
    """
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    if class_names is None:
        class_names = [f"class{i}" for i in range(num_labels)]
    if len(class_names) != num_labels:
        raise ValueError(f"class_names must have length {num_labels}, got {len(class_names)}")

    # Benchmark texts (deterministic sample)
    df_bench = df_v.sample(n=min(bench_samples, len(df_v)), random_state=123).reset_index(drop=True)
    bench_texts = df_bench[text_col].astype(str).tolist()

    # Evaluation set
    texts_v = df_v[text_col].astype(str).tolist()
    y_true = df_v[label_col].astype(int).to_numpy()

    rows = []

    for name in sorted(os.listdir(outputs_dir)):
        model_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(model_dir):
            continue

        # loadable model dirs only
        try:
            tokenizer, model, forced_device = _load_model_from_dir(model_dir, num_labels=num_labels)
        except Exception:
            continue

        used_device = forced_device if forced_device is not None else device

        # Predict + metrics
        y_pred = _predict_labels(model, tokenizer, texts_v, used_device, max_length, batch_size)
        metrics = _compute_metrics(y_true, y_pred, num_labels=num_labels)

        # Confusion matrix plot per model
        if save_confusion_matrices:
            cm_path = os.path.join(model_dir, "confusion_matrix.png")
            plot_confusion_matrix_counts_and_percents(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                out_png=cm_path,
                normalize="true",
                title=f"Confusion Matrix: {name}",
                show=show_confusion_matrices,
            )

        # Size & speed
        size_mb = _dir_size_mb(model_dir)
        ms_per_sample = _benchmark_ms_per_sample(model, tokenizer, bench_texts, used_device, max_length, batch_size)

        row = {
            "model_name": name,
            "model_size_mb": size_mb,
            "inference_ms_per_sample": ms_per_sample,
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "accuracy": metrics["accuracy"],
            "device_used": used_device,
            "path": model_dir,
        }
        for c in range(num_labels):
            row[f"f1_c{c}"] = _safe_float(metrics.get(f"f1_c{c}"))
            row[f"acc_c{c}"] = _safe_float(metrics.get(f"acc_c{c}"))

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No loadable model versions found in {outputs_dir}. "
            "Expected subfolders with config.json (HF) or meta.json (INT8 dynamic)."
        )

    df_out = pd.DataFrame(rows)

    # Column order: core comparison like your Table 2, then per-class diagnostics
    core = [
        "model_name",
        "model_size_mb",
        "inference_ms_per_sample",
        "macro_f1",
        "balanced_accuracy",
        "kappa",
        "accuracy",
        "device_used",
        "path",
    ]
    diag = []
    for c in range(num_labels):
        diag += [f"f1_c{c}", f"acc_c{c}"]

    cols = [c for c in core if c in df_out.columns] + [c for c in diag if c in df_out.columns]
    df_out = df_out[cols].sort_values("macro_f1", ascending=False, na_position="last").reset_index(drop=True)

    # Print nicely
    print("\n=== MODEL COMPARISON (CPU) ===")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)
    print(df_out.to_string(index=False))

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df_out.to_csv(save_csv_path, index=False)
        print(f"\nSaved CSV to: {save_csv_path}")

    return df_out


# ----------------------------
# Usage 
# ----------------------------
if __name__ == "__main__":
    outputs_dir = "outputs_part2"
    df_v = pd.read_csv("data/validation.csv")  # or your validation dataframe
    evaluate_all_versions_from_outputs(
         outputs_dir=outputs_dir,
         df_v=df_v,
         text_col="text",
         label_col="label",
         num_labels=6,
         device="cpu",
         max_length=128,
         batch_size=32,
         save_csv_path=os.path.join(outputs_dir, "model_comparison.csv"),
         class_names=["class0","class1","class2","class3","class4","class5"],
         save_confusion_matrices=True,
         show_confusion_matrices=True
    )
  
