# full_pipeline_with_earlystop_compression_supersample.py
# COMPRESSION 1: PRUNING + RECOVERY FINETUNE
# COMPRESSION 2: DYNAMIC INT8 QUANTIZATION (CPU)

import os
import json
import time
import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)


# -------------------------
# Version-proof TrainingArguments
# -------------------------
def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in sig and "eval_strategy" in sig:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in sig and "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


# -------------------------
# Dataset
# -------------------------
class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, max_length: int = 128):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        return enc


# -------------------------
# Weighted-loss Trainer (new Trainer-safe)
# -------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)

    labels = np.asarray(labels)
    preds = np.asarray(preds)
    num_labels = int(max(labels.max(), preds.max())) + 1

    per_class = {}
    for c in range(num_labels):
        mask = labels == c
        per_class[f"acc_c{c}"] = float((preds[mask] == labels[mask]).mean()) if mask.sum() else float("nan")

    out = {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "kappa": float(kappa),
    }
    out.update(per_class)
    return out


# -------------------------
# Supersampling (oversampling by duplication)
# -------------------------
def parse_int_list(s: str) -> List[int]:
    """
    Accepts formats like:
      "[1,1,1,1,2]" or "1,1,1,1,2"
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(x) for x in parts]


def supersample_train_df(train_df: pd.DataFrame, label_col: str, factors: List[int], seed: int) -> pd.DataFrame:
    """
    For each class c, duplicate that class rows 'factors[c]' times total.
    factor=1 -> keep original
    factor=2 -> original + one extra copy
    factor=3 -> original + two extra copies, etc.
    """
    y = train_df[label_col].astype(int)
    num_labels = int(y.nunique())

    # Allow missing classes in split: infer num_labels from max label
    max_label = int(y.max())
    inferred_num_labels = max_label + 1

    if len(factors) != inferred_num_labels:
        raise ValueError(
            f"supersample_factors length ({len(factors)}) must equal num_labels ({inferred_num_labels}). "
            f"Got factors={factors}"
        )
    if any(f < 1 for f in factors):
        raise ValueError("All supersample factors must be >= 1.")

    rng = np.random.default_rng(seed)

    pieces = []
    for c, f in enumerate(factors):
        cls_df = train_df[train_df[label_col].astype(int) == c]
        if len(cls_df) == 0:
            continue
        if f == 1:
            pieces.append(cls_df)
        else:
            # duplicate by concatenation (shuffle each duplicated copy for variety)
            reps = [cls_df]
            for _ in range(f - 1):
                reps.append(cls_df.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1e9))))
            pieces.append(pd.concat(reps, axis=0))

    out = pd.concat(pieces, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


# -------------------------
# Plot losses
# -------------------------
def plot_losses_from_trainer(trainer: Trainer, out_png: str, show: bool = False):
    log = trainer.state.log_history
    train_epoch_loss = {}
    eval_epoch_loss = {}

    for item in log:
        if "epoch" not in item:
            continue
        ep = float(item["epoch"])
        if "loss" in item and "eval_loss" not in item:
            train_epoch_loss[ep] = float(item["loss"])
        if "eval_loss" in item:
            eval_epoch_loss[ep] = float(item["eval_loss"])

    epochs_train = sorted(train_epoch_loss.keys())
    epochs_eval = sorted(eval_epoch_loss.keys())

    plt.figure()
    if epochs_train:
        plt.plot(epochs_train, [train_epoch_loss[e] for e in epochs_train], marker="o", label="train_loss")
    if epochs_eval:
        plt.plot(epochs_eval, [eval_epoch_loss[e] for e in epochs_eval], marker="o", label="val_loss (eval_loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


def model_disk_size_mb(model: nn.Module, tmp_path: str = "tmp_state_dict.pt") -> float:
    torch.save(model.state_dict(), tmp_path)
    mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return float(mb)


# -------------------------
# Compression
# -------------------------
def apply_global_magnitude_pruning(model: nn.Module, amount: float = 0.35) -> nn.Module:
    params = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            params.append((m, "weight"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for (m, name) in params:
        prune.remove(m, name)
    return model


@torch.no_grad()
def linear_sparsity(model: nn.Module) -> float:
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            zeros += (w == 0).sum().item()
            total += w.numel()
    return zeros / max(total, 1)


def dynamic_int8_quantize(model: nn.Module) -> nn.Module:
    model_cpu = model.to("cpu").eval()
    qmodel = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    return qmodel


# -------------------------
# Training wrapper
# -------------------------
@dataclass
class TrainResult:
    model_name: str
    saved_dir: str
    best_metric: float
    metrics: Dict[str, float]
    size_mb: float
    loss_plot_path: str


def train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    num_labels: int,
    out_root: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_length: int,
    patience: int,
    use_weighted_loss: bool,
    device: str,
    show_plots: bool,
) -> TrainResult:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_ds = CSVDataset(train_df, tok, text_col, label_col, max_length=max_length)
    val_ds = CSVDataset(val_df, tok, text_col, label_col, max_length=max_length)
    collator = DataCollatorWithPadding(tok)

    class_counts = train_df[label_col].value_counts().sort_index().reindex(range(num_labels), fill_value=0).values
    inv = 1.0 / torch.tensor(class_counts + 1e-9, dtype=torch.float)  # +eps to avoid div by 0
    class_weights = inv / inv.sum()

    save_dir = os.path.join(out_root, model_name.replace("/", "__"))
    os.makedirs(save_dir, exist_ok=True)

    targs = make_training_args(
        output_dir=save_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        fp16=("cuda" in device and torch.cuda.is_available()),
    )

    if use_weighted_loss:
        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)

    loss_png = os.path.join(save_dir, "loss_curve.png")
    plot_losses_from_trainer(trainer, loss_png, show=show_plots)

    size_mb = model_disk_size_mb(trainer.model)
    best_metric = float(metrics.get("eval_macro_f1", float("nan")))

    metrics_clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
    return TrainResult(
        model_name=model_name,
        saved_dir=save_dir,
        best_metric=best_metric,
        metrics=metrics_clean,
        size_mb=size_mb,
        loss_plot_path=loss_png,
    )


# -------------------------
# Main pipeline
#  sample usage, adjust of needed
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--out_dir", type=str, default="./outputs_part2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--no_weighted_loss", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show_plots", action="store_true")

    p.add_argument(
        "--supersample_factors",
        type=str,
        default="1,1,4,2,3,8",
        help='e.g. "1,1,1,1,1,1" (no oversample) or "1,1,1,1,1,2" (double class 5)'
    )

    # Model Compression
    p.add_argument("--prune_amount", type=float, default=0.35)
    p.add_argument("--prune_recover_epochs", type=int, default=1)

    args, _ = p.parse_known_args(argv)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    num_labels = int(df[args.label_col].nunique())

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df[args.label_col],
    )

    # Apply supersampling ONLY to training split
    if args.supersample_factors is not None:
        factors = parse_int_list(args.supersample_factors)
        train_df = supersample_train_df(train_df, args.label_col, factors=factors, seed=args.seed)
        print("Applied supersampling factors:", factors)
        print("New train label counts:\n", train_df[args.label_col].value_counts().sort_index())

    candidates = [
        "roberta-base",
        "cardiffnlp/twitter-roberta-base",
        "distilroberta-base",
    ]

    results: List[TrainResult] = []
    for m in candidates:
        print(f"\n=== Training {m} ===")
        r = train_one_model(
            model_name=m,
            train_df=train_df,
            val_df=val_df,
            text_col=args.text_col,
            label_col=args.label_col,
            num_labels=num_labels,
            out_root=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            patience=args.patience,
            use_weighted_loss=(not args.no_weighted_loss),
            device=args.device,
            show_plots=args.show_plots,
        )
        results.append(r)
        print("Saved:", r.saved_dir)
        print("Loss plot:", r.loss_plot_path)
        print("Size(MB):", round(r.size_mb, 2))
        print("Eval macro-F1:", round(r.best_metric, 6))

    best = max(results, key=lambda x: x.best_metric)
    print("\n=== BEST MODEL ===")
    print("Best:", best.model_name)
    print("Dir :", best.saved_dir)

    # -------------------------
    # COMPRESSION 1: PRUNING + RECOVERY FINETUNE
    # -------------------------
    print("\n=== COMPRESSION 1: PRUNING + RECOVERY FINETUNE ===")
    best_tok = AutoTokenizer.from_pretrained(best.saved_dir, use_fast=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(best.saved_dir)

    pruned_model = apply_global_magnitude_pruning(best_model, amount=args.prune_amount)
    sp = linear_sparsity(pruned_model)
    print(f"Pruned linear sparsity: {sp:.3f}")

    pruned_dir = os.path.join(args.out_dir, "BEST_PRUNED")
    os.makedirs(pruned_dir, exist_ok=True)

    train_ds = CSVDataset(train_df, best_tok, args.text_col, args.label_col, max_length=args.max_length)
    val_ds = CSVDataset(val_df, best_tok, args.text_col, args.label_col, max_length=args.max_length)
    collator = DataCollatorWithPadding(best_tok)

    class_counts = train_df[args.label_col].value_counts().sort_index().reindex(range(num_labels), fill_value=0).values
    inv = 1.0 / torch.tensor(class_counts + 1e-9, dtype=torch.float)
    class_weights = inv / inv.sum()

    pruned_args = make_training_args(
        output_dir=pruned_dir,
        num_train_epochs=args.prune_recover_epochs,
        learning_rate=min(args.lr, 1e-5),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
        fp16=("cuda" in args.device and torch.cuda.is_available()),
    )

    if args.no_weighted_loss:
        pruned_trainer = Trainer(
            model=pruned_model,
            args=pruned_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=best_tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
    else:
        pruned_trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=pruned_model,
            args=pruned_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=best_tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

    pruned_trainer.train()
    pruned_metrics = pruned_trainer.evaluate()

    pruned_trainer.save_model(pruned_dir)
    best_tok.save_pretrained(pruned_dir)

    pruned_loss_png = os.path.join(pruned_dir, "loss_curve.png")
    plot_losses_from_trainer(pruned_trainer, pruned_loss_png, show=args.show_plots)

    pruned_size = model_disk_size_mb(pruned_trainer.model)
    print("Pruned dir:", pruned_dir)
    print("Pruned size(MB):", round(pruned_size, 2))
    print("Pruned metrics:", {k: round(float(v), 6) for k, v in pruned_metrics.items() if isinstance(v, (int, float, np.floating))})

    # -------------------------
    # COMPRESSION 2: DYNAMIC INT8 QUANTIZATION (CPU)
    # -------------------------
    print("\n=== COMPRESSION 2: DYNAMIC INT8 QUANTIZATION (CPU) ===")
    best_model_clean = AutoModelForSequenceClassification.from_pretrained(best.saved_dir)
    qmodel = dynamic_int8_quantize(best_model_clean)

    texts = val_df[args.text_col].astype(str).tolist()
    labels = val_df[args.label_col].astype(int).to_numpy()

    preds_all = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = best_tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = qmodel(**enc).logits
        preds_all.append(logits.argmax(dim=-1).cpu().numpy())

    preds = np.concatenate(preds_all, axis=0)

    q_metrics = {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "kappa": float(cohen_kappa_score(labels, preds)),
    }
    for c in range(num_labels):
        mask = labels == c
        q_metrics[f"acc_c{c}"] = float((preds[mask] == labels[mask]).mean()) if mask.sum() else float("nan")

    q_size = model_disk_size_mb(qmodel)

    quant_dir = os.path.join(args.out_dir, "BEST_INT8_CPU")
    os.makedirs(quant_dir, exist_ok=True)
    torch.save(qmodel.state_dict(), os.path.join(quant_dir, "pytorch_model.bin"))
    best_tok.save_pretrained(quant_dir)
    with open(os.path.join(quant_dir, "meta.json"), "w") as f:
        json.dump({"base_model_dir": best.saved_dir, "note": "dynamic int8 quantized (CPU) Linear layers"}, f, indent=2)

    print("Quant dir:", quant_dir)
    print("Quant size(MB):", round(q_size, 2))
    print("Quant metrics:", {k: round(float(v), 6) for k, v in q_metrics.items()})

    # Summary JSON
    summary_path = os.path.join(args.out_dir, "summary.json")
    summary = {
        "best_model": best.model_name,
        "best_dir": best.saved_dir,
        "best_metrics": best.metrics,
        "pruned_dir": pruned_dir,
        "pruned_metrics": {k: float(v) for k, v in pruned_metrics.items() if isinstance(v, (int, float, np.floating))},
        "pruned_sparsity": sp,
        "quant_dir": quant_dir,
        "quant_metrics": q_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print("Summary:", summary_path)
    print("All outputs in:", args.out_dir)


if __name__ == "__main__":
    main()


#python full_pipeline_with_earlystop_compression_supersample.py --supersample_factors "1,1,1,1,1,1"
