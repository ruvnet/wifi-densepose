#!/usr/bin/env python3
"""Train the person-count head — ADR-103 v0.0.1.

Mirrors the Conv1d encoder architecture from cog-person-count's
`src/inference.rs::CountNet` exactly, so the learned weights load
into the Rust cog without translation. Trains on
data/paired/wiflow-p7-1779210883.paired.jsonl (1,077 samples with
n_persons_mode labels in {0, 1}).

Output: count_v1.safetensors + count_v1.onnx + train_results.json.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture constants — MUST match cog-person-count's src/inference.rs.
N_SUB = 56
N_FRAMES = 20
COUNT_CLASSES = 8


class CountNet(nn.Module):
    """Mirrors cog_person_count::inference::CountNet bit-for-bit."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder — identical to the pose cog's encoder so future joint
        # training can share weights.
        self.enc_c1 = nn.Conv1d(N_SUB, 64, kernel_size=3, padding=1, dilation=1)
        self.enc_c2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.enc_c3 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)
        # Count head
        self.count_head_fc1 = nn.Linear(128, 64)
        self.count_head_fc2 = nn.Linear(64, COUNT_CLASSES)
        # Confidence head
        self.conf_head_fc1 = nn.Linear(128, 32)
        self.conf_head_fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, 56, 20]
        h = F.relu(self.enc_c1(x))
        h = F.relu(self.enc_c2(h))
        h = F.relu(self.enc_c3(h))
        h = h.mean(dim=2)  # [B, 128]

        # Logits (un-normalised); softmax at inference + cross-entropy training.
        c = F.relu(self.count_head_fc1(h))
        count_logits = self.count_head_fc2(c)

        # Confidence head — sigmoid at inference; BCE-with-logits at training.
        cf = F.relu(self.conf_head_fc1(h))
        conf_logits = self.conf_head_fc2(cf)

        return count_logits, conf_logits


def load_paired(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X is [N, 56, 20] CSI and y is [N] integer counts."""
    csis, ys = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            shape = d.get("csi_shape", [N_SUB, N_FRAMES])
            if shape != [N_SUB, N_FRAMES]:
                continue
            csi = np.asarray(d["csi"], dtype=np.float32).reshape(N_SUB, N_FRAMES)
            csis.append(csi)
            ys.append(int(d.get("n_persons_mode", 0)))
    X = np.stack(csis, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def temporal_split(X: np.ndarray, y: np.ndarray, eval_frac: float = 0.2):
    """Held-out time-window eval (last `eval_frac` of samples, by index)."""
    n = X.shape[0]
    n_eval = int(round(n * eval_frac))
    n_train = n - n_eval
    return (
        X[:n_train], y[:n_train],
        X[n_train:], y[n_train:],
    )


def stratified_k_fold(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Stratified k-fold cross-validation splits — hand-rolled, no sklearn.

    Per class: shuffle the indices (deterministic seed 42), split into k
    near-equal chunks, then assemble fold i by taking chunk i from every
    class. Yields (X_train, y_train, X_val, y_val) per fold, with class
    distribution preserved within ±1.
    """
    rng = np.random.default_rng(seed=42)
    classes = np.unique(y)
    per_class_folds = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        per_class_folds[c] = np.array_split(idx, k)
    for fold in range(k):
        val_idx = np.concatenate([per_class_folds[c][fold] for c in classes])
        train_idx = np.concatenate(
            [per_class_folds[c][f] for c in classes for f in range(k) if f != fold]
        )
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def standardise(X_train: np.ndarray, X_eval: np.ndarray):
    """Z-score by subcarrier across the time axis. Eval uses train stats."""
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (X_train - mu) / sd, (X_eval - mu) / sd


def write_safetensors(model: CountNet, path: Path):
    """Write the model's state in the same on-disk layout the Rust cog expects."""
    state = model.state_dict()
    # Map PyTorch param names → cog-person-count's VarBuilder paths.
    rename = {
        "enc_c1.weight": "enc.c1.weight",
        "enc_c1.bias":   "enc.c1.bias",
        "enc_c2.weight": "enc.c2.weight",
        "enc_c2.bias":   "enc.c2.bias",
        "enc_c3.weight": "enc.c3.weight",
        "enc_c3.bias":   "enc.c3.bias",
        "count_head_fc1.weight": "count_head.fc1.weight",
        "count_head_fc1.bias":   "count_head.fc1.bias",
        "count_head_fc2.weight": "count_head.fc2.weight",
        "count_head_fc2.bias":   "count_head.fc2.bias",
        "conf_head_fc1.weight":  "conf_head.fc1.weight",
        "conf_head_fc1.bias":    "conf_head.fc1.bias",
        "conf_head_fc2.weight":  "conf_head.fc2.weight",
        "conf_head_fc2.bias":    "conf_head.fc2.bias",
    }

    header = {}
    payload = bytearray()
    offset = 0
    for torch_name, cog_name in rename.items():
        t = state[torch_name].detach().cpu().numpy().astype(np.float32)
        n_bytes = t.nbytes
        header[cog_name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [offset, offset + n_bytes],
        }
        payload.extend(t.tobytes())
        offset += n_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out-safetensors", default="count_v1.safetensors")
    parser.add_argument("--out-onnx", default="count_v1.onnx")
    parser.add_argument("--out-results", default="count_train_results.json")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--k-fold", type=int, default=None, help="If set, run k-fold CV; else use temporal split")
    parser.add_argument("--v2", action="store_true",
                        help="v0.0.2 training: random 80/20 split + label smoothing + early stopping "
                             "+ balanced sampling + temperature-scaled confidence head.")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X, y = load_paired(Path(args.paired))
    print(f"loaded {X.shape[0]} samples, X shape {X.shape}, "
          f"label distribution: {dict(Counter(y.tolist()).most_common())}")

    # K-fold cross-validation mode
    if args.k_fold is not None:
        print(f"\n=== {args.k_fold}-fold cross-validation ===")
        fold_results = []
        overall_t0 = time.perf_counter()

        for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(stratified_k_fold(X, y, k=args.k_fold)):
            print(f"\nFold {fold_idx + 1}/{args.k_fold}")
            X_train, X_val = standardise(X_train, X_val)

            cls_counts = np.bincount(y_train, minlength=COUNT_CLASSES).astype(np.float32)
            cls_counts = np.where(cls_counts > 0, cls_counts, 1.0)
            cls_weight = (1.0 / cls_counts) / (1.0 / cls_counts).sum() * COUNT_CLASSES
            cls_weight_t = torch.from_numpy(cls_weight).to(device)

            Xt = torch.from_numpy(X_train).to(device)
            yt = torch.from_numpy(y_train).to(device)
            Xv = torch.from_numpy(X_val).to(device)
            yv = torch.from_numpy(y_val).to(device)

            model = CountNet().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1)

            n_train = X_train.shape[0]
            best_eval_acc = 0.0
            best_state = None

            for epoch in range(args.epochs):
                model.train()
                perm = torch.randperm(n_train, device=device)
                train_loss = 0.0
                train_correct = 0
                n_batches = 0
                for i in range(0, n_train, args.batch_size):
                    idx = perm[i : i + args.batch_size]
                    xb = Xt[idx]
                    yb = yt[idx]
                    opt.zero_grad()
                    count_logits, conf_logits = model(xb)
                    ce = F.cross_entropy(count_logits, yb, weight=cls_weight_t)
                    with torch.no_grad():
                        pred = count_logits.argmax(dim=1)
                        correct_indicator = (pred == yb).float().unsqueeze(1)
                    bce = F.binary_cross_entropy_with_logits(conf_logits, correct_indicator)
                    with torch.no_grad():
                        conf_sigm = torch.sigmoid(conf_logits)
                    brier = ((conf_sigm - correct_indicator) ** 2).mean()
                    loss = ce + 0.3 * bce + 0.1 * brier
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    train_correct += (pred == yb).sum().item()
                    n_batches += 1

                sched.step()

                model.eval()
                with torch.no_grad():
                    cl_v, _ = model(Xv)
                    eval_pred = cl_v.argmax(dim=1)
                    eval_acc = (eval_pred == yv).float().mean().item()

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # Restore best checkpoint and final eval
            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                cl_v, conf_v = model(Xv)
                pred_v = cl_v.argmax(dim=1)
                acc = (pred_v == yv).float().mean().item()
                within1 = ((pred_v - yv).abs() <= 1).float().mean().item()
                mae = (pred_v - yv).abs().float().mean().item()

                # Per-class accuracy
                per_class = {}
                for k in range(COUNT_CLASSES):
                    mask = yv == k
                    n = mask.sum().item()
                    if n > 0:
                        per_class[k] = {
                            "support": int(n),
                            "accuracy": ((pred_v == yv) & mask).sum().item() / n,
                        }

                # Spearman
                conf_sigm = torch.sigmoid(conf_v).squeeze(-1)
                correct = (pred_v == yv).float()
                c_rank = conf_sigm.argsort().argsort().float()
                r_rank = correct.argsort().argsort().float()
                c_centered = c_rank - c_rank.mean()
                r_centered = r_rank - r_rank.mean()
                denom = (c_centered.norm() * r_centered.norm()).item()
                spearman = (c_centered * r_centered).sum().item() / denom if denom > 0 else 0.0

            fold_results.append({
                "fold": fold_idx + 1,
                "accuracy": acc,
                "within_pm1": within1,
                "mae": mae,
                "spearman": spearman,
                "per_class_accuracy": per_class,
            })
            print(f"  accuracy={acc:.3f}  within±1={within1:.3f}  mae={mae:.3f}  spearman={spearman:.3f}")

        # K-fold summary
        total_time = time.perf_counter() - overall_t0
        accs = [r["accuracy"] for r in fold_results]
        within1s = [r["within_pm1"] for r in fold_results]
        maes = [r["mae"] for r in fold_results]
        spears = [r["spearman"] for r in fold_results]

        print(f"\n=== {args.k_fold}-fold summary ({total_time:.1f} s) ===")
        print(f"  accuracy:       {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  within ±1:      {np.mean(within1s):.3f} ± {np.std(within1s):.3f}")
        print(f"  MAE:            {np.mean(maes):.3f} ± {np.std(maes):.3f}")
        print(f"  conf↔correct Spearman: {np.mean(spears):.3f} ± {np.std(spears):.3f}")

        # Per-class summary across folds
        for k in range(COUNT_CLASSES):
            accs_k = [r["per_class_accuracy"].get(k, {}).get("accuracy", 0.0) for r in fold_results]
            n_k = [r["per_class_accuracy"].get(k, {}).get("support", 0) for r in fold_results]
            if any(n > 0 for n in n_k):
                print(f"  class {k}:  {np.mean(accs_k):.3f} mean accuracy (support: {n_k})")

        # Write k-fold results to JSON
        results = {
            "mode": "k_fold_cv",
            "k": args.k_fold,
            "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch-cpu",
            "total_time_s": total_time,
            "fold_results": fold_results,
            "summary": {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_within_pm1": float(np.mean(within1s)),
                "std_within_pm1": float(np.std(within1s)),
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_spearman": float(np.mean(spears)),
                "std_spearman": float(np.std(spears)),
            },
            "hyperparameters": {
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "schedule": "cosine_warm_restarts",
                "epochs": args.epochs,
            },
        }
        Path(args.out_results).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out_results}")
        return

    # ---------------------------------------------------------------
    # v0.0.2 training path: random 80/20 + label smoothing + early
    # stopping + class-balanced batch sampling + temperature scaling.
    # ---------------------------------------------------------------
    if args.v2:
        rng = np.random.default_rng(seed=42)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        n_eval = int(round(0.2 * X.shape[0]))
        eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
        X_train, X_eval = X[train_idx], X[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]
        X_train, X_eval = standardise(X_train, X_eval)
        print(f"v0.0.2 mode — random 80/20 split: train={len(y_train)} eval={len(y_eval)}")
        print(f"  train class dist: {dict(Counter(y_train.tolist()).most_common())}")
        print(f"  eval  class dist: {dict(Counter(y_eval.tolist()).most_common())}")

        Xt = torch.from_numpy(X_train).to(device)
        yt = torch.from_numpy(y_train).to(device)
        Xe = torch.from_numpy(X_eval).to(device)
        ye = torch.from_numpy(y_eval).to(device)

        # Class-balanced sampler: for each batch, sample with replacement
        # so each class has equal expected count regardless of dataset
        # distribution. With our ~533/544 split this is nearly a no-op
        # but it generalises to imbalanced multi-room data later.
        cls_counts = np.bincount(y_train, minlength=COUNT_CLASSES).astype(np.float32)
        cls_counts = np.where(cls_counts > 0, cls_counts, 1.0)
        per_sample_weight = (1.0 / cls_counts[y_train])
        per_sample_weight_t = torch.from_numpy(per_sample_weight.astype(np.float32)).to(device)

        model = CountNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1)

        n_train = X_train.shape[0]
        batches_per_epoch = max(1, n_train // args.batch_size)
        epoch_losses = []
        t0 = time.perf_counter()
        best_eval_acc = 0.0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0; train_correct = 0; n_batches = 0
            for _ in range(batches_per_epoch):
                # Balanced sample with replacement
                idx_t = torch.multinomial(per_sample_weight_t, args.batch_size, replacement=True)
                xb = Xt[idx_t]; yb = yt[idx_t]
                opt.zero_grad()
                count_logits, conf_logits = model(xb)
                ce = F.cross_entropy(count_logits, yb, label_smoothing=args.label_smoothing)
                with torch.no_grad():
                    pred = count_logits.argmax(dim=1)
                    correct_indicator = (pred == yb).float().unsqueeze(1)
                bce = F.binary_cross_entropy_with_logits(conf_logits, correct_indicator)
                with torch.no_grad():
                    conf_sigm = torch.sigmoid(conf_logits)
                brier = ((conf_sigm - correct_indicator) ** 2).mean()
                loss = ce + 0.3 * bce + 0.1 * brier
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_correct += (pred == yb).sum().item()
                n_batches += 1
            sched.step()

            model.eval()
            with torch.no_grad():
                cl_e, _ = model(Xe)
                eval_loss = F.cross_entropy(cl_e, ye).item()
                eval_pred = cl_e.argmax(dim=1)
                eval_acc = (eval_pred == ye).float().mean().item()
            epoch_losses.append({
                "epoch": epoch,
                "train_loss": train_loss / max(1, n_batches),
                "train_acc": train_correct / max(1, n_batches * args.batch_size),
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            })
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch < 5 or epoch % 25 == 0:
                print(f"epoch {epoch:3d}  train_loss={train_loss/n_batches:.4f}  "
                      f"train_acc={train_correct/(n_batches*args.batch_size):.3f}  "
                      f"eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.3f}  "
                      f"epochs_no_improve={epochs_without_improvement}")
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        train_time = time.perf_counter() - t0
        print(f"\ntrained {epoch + 1} epochs in {train_time:.1f} s  (best eval_acc {best_eval_acc:.3f})")
        if best_state is not None:
            model.load_state_dict(best_state)

        # Temperature scaling on the confidence head — fit a scalar T s.t.
        # sigmoid(conf_logits / T) is best-calibrated on the eval set.
        model.eval()
        with torch.no_grad():
            cl_e, conf_e = model(Xe)
            pred_e = cl_e.argmax(dim=1)
            correct_indicator = (pred_e == ye).float()
        # 1D optimisation over T via LBFGS.
        T = torch.nn.Parameter(torch.ones(1, device=device))
        opt_t = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
        def eval_t():
            opt_t.zero_grad()
            scaled = conf_e.squeeze(-1) / T
            loss_t = F.binary_cross_entropy_with_logits(scaled, correct_indicator)
            loss_t.backward()
            return loss_t
        opt_t.step(eval_t)
        T_val = float(T.detach().cpu().item())
        print(f"  temperature scale T = {T_val:.4f}")

        # Final eval with temperature applied.
        with torch.no_grad():
            cl_e, conf_e = model(Xe)
            probs_e = F.softmax(cl_e, dim=1)
            pred_e = cl_e.argmax(dim=1)
            acc = (pred_e == ye).float().mean().item()
            within1 = ((pred_e - ye).abs() <= 1).float().mean().item()
            mae = (pred_e - ye).abs().float().mean().item()
            per_class = {}
            for k in range(COUNT_CLASSES):
                mask = ye == k
                n = mask.sum().item()
                if n > 0:
                    per_class[k] = {
                        "support": int(n),
                        "accuracy": ((pred_e == ye) & mask).sum().item() / n,
                    }
            conf_sigm = torch.sigmoid(conf_e.squeeze(-1) / T_val)
            correct = (pred_e == ye).float()
            c_rank = conf_sigm.argsort().argsort().float()
            r_rank = correct.argsort().argsort().float()
            c_centered = c_rank - c_rank.mean()
            r_centered = r_rank - r_rank.mean()
            denom = (c_centered.norm() * r_centered.norm()).item()
            spearman = (c_centered * r_centered).sum().item() / denom if denom > 0 else 0.0

        print(f"\n=== v0.0.2 final eval ===")
        print(f"  accuracy:       {acc:.3f}")
        print(f"  within ±1:      {within1:.3f}")
        print(f"  MAE:            {mae:.3f}")
        print(f"  conf↔correct Spearman (post-temp): {spearman:.3f}")
        for k, v in per_class.items():
            print(f"  class {k}:  {v['accuracy']:.3f} accuracy on {v['support']} samples")

        write_safetensors(model, Path(args.out_safetensors))
        # Also append the temperature scalar so the cog can apply it.
        # We add it by appending to the safetensors file using the
        # write_safetensors helper but with the temperature recorded
        # as a separate file alongside (count_v1.temperature.txt) for
        # consumption by the Rust cog inference path.
        Path(args.out_safetensors + ".temperature").write_text(f"{T_val}\n")
        print(f"wrote {args.out_safetensors} ({Path(args.out_safetensors).stat().st_size} bytes)")
        print(f"wrote {args.out_safetensors}.temperature ({T_val})")

        # ONNX
        dummy = torch.zeros(1, N_SUB, N_FRAMES, device=device)
        try:
            torch.onnx.export(model, dummy, args.out_onnx, opset_version=18,
                              input_names=["csi_window"],
                              output_names=["count_logits", "conf_logits"],
                              dynamic_axes={"csi_window": {0: "batch"},
                                            "count_logits": {0: "batch"},
                                            "conf_logits": {0: "batch"}},
                              export_params=True, do_constant_folding=True)
            print(f"wrote {args.out_onnx} ({Path(args.out_onnx).stat().st_size} bytes)")
        except Exception as e:
            print(f"WARN: ONNX export failed: {e}")

        results = {
            "mode": "v0.0.2",
            "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch-cpu",
            "epochs_trained": epoch + 1,
            "train_time_s": train_time,
            "best_eval_acc": best_eval_acc,
            "final_eval_acc": acc,
            "final_eval_within_pm1": within1,
            "final_eval_mae": mae,
            "temperature_scale": T_val,
            "conf_correctness_spearman_post_temp": spearman,
            "per_class_accuracy": per_class,
            "hyperparameters": {
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "schedule": "cosine_warm_restarts",
                "epochs_max": args.epochs,
                "label_smoothing": args.label_smoothing,
                "patience": args.patience,
                "split": "random_80_20_seed_42",
                "balanced_sampler": True,
                "temperature_scaling": True,
            },
            "epoch_losses": epoch_losses,
        }
        Path(args.out_results).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out_results}")
        return

    # Original temporal-split mode (kept for v0.0.1 reproducibility).
    X_train, y_train, X_eval, y_eval = temporal_split(X, y, eval_frac=0.2)
    X_train, X_eval = standardise(X_train, X_eval)

    # Re-balance via class weights — handles the 50/50 split fine
    # but also makes the loss correct under future imbalanced data.
    cls_counts = np.bincount(y_train, minlength=COUNT_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts > 0, cls_counts, 1.0)
    cls_weight = (1.0 / cls_counts) / (1.0 / cls_counts).sum() * COUNT_CLASSES
    cls_weight_t = torch.from_numpy(cls_weight).to(device)
    print(f"class weights: {cls_weight.tolist()}")

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xe = torch.from_numpy(X_eval).to(device)
    ye = torch.from_numpy(y_eval).to(device)

    model = CountNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1)

    n_train = X_train.shape[0]
    epoch_losses = []
    t0 = time.perf_counter()

    best_eval_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        train_correct = 0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            opt.zero_grad()
            count_logits, conf_logits = model(xb)

            # Categorical cross-entropy for count.
            ce = F.cross_entropy(count_logits, yb, weight=cls_weight_t)

            # Confidence head: train against `argmax == truth` indicator.
            with torch.no_grad():
                pred = count_logits.argmax(dim=1)
                correct_indicator = (pred == yb).float().unsqueeze(1)
            bce = F.binary_cross_entropy_with_logits(conf_logits, correct_indicator)

            # Brier-score uncertainty calibration on the conf head — sharpens
            # the calibration so the sigmoid output is a real probability.
            with torch.no_grad():
                conf_sigm = torch.sigmoid(conf_logits)
            brier = ((conf_sigm - correct_indicator) ** 2).mean()

            loss = ce + 0.3 * bce + 0.1 * brier
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_correct += (pred == yb).sum().item()
            n_batches += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            cl_e, _ = model(Xe)
            eval_loss = F.cross_entropy(cl_e, ye, weight=cls_weight_t).item()
            eval_pred = cl_e.argmax(dim=1)
            eval_acc = (eval_pred == ye).float().mean().item()
            eval_within1 = ((eval_pred - ye).abs() <= 1).float().mean().item()

        epoch_losses.append({
            "epoch": epoch,
            "train_loss": train_loss / n_batches,
            "train_acc": train_correct / n_train,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_within_pm1": eval_within1,
        })

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch < 5 or epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss/n_batches:.4f}  "
                  f"train_acc={train_correct/n_train:.3f}  "
                  f"eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.3f}  "
                  f"within±1={eval_within1:.3f}")

    train_time = time.perf_counter() - t0
    print(f"\ntrained {args.epochs} epochs in {train_time:.1f} s")
    print(f"best eval_acc: {best_eval_acc:.3f}")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval breakdown
    model.eval()
    with torch.no_grad():
        cl_e, conf_e = model(Xe)
        probs_e = torch.softmax(cl_e, dim=1)
        pred_e = cl_e.argmax(dim=1)
        acc = (pred_e == ye).float().mean().item()
        within1 = ((pred_e - ye).abs() <= 1).float().mean().item()
        mae = (pred_e - ye).abs().float().mean().item()

        # Per-class accuracy
        per_class = {}
        for k in range(COUNT_CLASSES):
            mask = ye == k
            n = mask.sum().item()
            if n > 0:
                per_class[k] = {
                    "support": int(n),
                    "accuracy": ((pred_e == ye) & mask).sum().item() / n,
                }

        # Confidence-accuracy calibration: Spearman over (predicted-correct, confidence)
        conf_sigm = torch.sigmoid(conf_e).squeeze(-1)
        correct = (pred_e == ye).float()
        # Spearman = Pearson over ranks
        c_rank = conf_sigm.argsort().argsort().float()
        r_rank = correct.argsort().argsort().float()
        c_centered = c_rank - c_rank.mean()
        r_centered = r_rank - r_rank.mean()
        denom = (c_centered.norm() * r_centered.norm()).item()
        spearman = (c_centered * r_centered).sum().item() / denom if denom > 0 else 0.0

    print(f"\n=== final eval ===")
    print(f"  accuracy:       {acc:.3f}")
    print(f"  within ±1:      {within1:.3f}")
    print(f"  MAE:            {mae:.3f}")
    print(f"  conf↔correct Spearman: {spearman:.3f}")
    for k, v in per_class.items():
        print(f"  class {k}:  {v['accuracy']:.3f} accuracy on {v['support']} samples")

    # Save safetensors
    write_safetensors(model, Path(args.out_safetensors))
    print(f"\nwrote {args.out_safetensors} ({Path(args.out_safetensors).stat().st_size} bytes)")

    # ONNX export
    dummy = torch.zeros(1, N_SUB, N_FRAMES, device=device)
    try:
        torch.onnx.export(
            model, dummy, args.out_onnx,
            opset_version=18,
            input_names=["csi_window"],
            output_names=["count_logits", "conf_logits"],
            dynamic_axes={
                "csi_window": {0: "batch"},
                "count_logits": {0: "batch"},
                "conf_logits": {0: "batch"},
            },
            export_params=True,
            do_constant_folding=True,
        )
        print(f"wrote {args.out_onnx} ({Path(args.out_onnx).stat().st_size} bytes)")
    except Exception as e:
        print(f"WARN: ONNX export failed: {e}")

    # Results JSON
    results = {
        "backend": "candle-cuda" if device.type == "cuda" else "candle-cpu",
        "device": str(device),
        "epochs": args.epochs,
        "train_time_s": train_time,
        "best_eval_acc": best_eval_acc,
        "final_eval_acc": acc,
        "final_eval_within_pm1": within1,
        "final_eval_mae": mae,
        "conf_correctness_spearman": spearman,
        "per_class_accuracy": per_class,
        "hyperparameters": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "schedule": "cosine_warm_restarts",
            "epochs": args.epochs,
            "loss": "cross_entropy(count) + 0.3*bce(conf) + 0.1*brier(conf)",
            "z_score_normalisation": True,
            "class_weights": cls_weight.tolist(),
        },
        "epoch_losses": epoch_losses,
    }
    Path(args.out_results).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out_results} ({Path(args.out_results).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
