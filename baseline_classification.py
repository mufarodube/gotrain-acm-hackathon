"""
Baseline for GEO-Bench classification (m-eurosat, m-pv4ger, ...).
Also the engine behind the agent's `run_experiment` tool, so every knob is a flag.

Usage:
    export GEO_BENCH_DIR=~/geobench_data
    python baseline_classification.py --task m-eurosat --epochs 10
    python baseline_classification.py --task m-pv4ger --img_size 160 --max_train 2000

Touches TRAIN and VALIDATION only. Never the test split.
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import geobench
import warnings

# Known geo-bench issue #28: m-eurosat band NAMES are wrong (pixel data is fine).
# We use all bands by position, so this warning only buries real errors.
warnings.filterwarnings("ignore", message=".*incorrect band names.*")


def get_task(name: str, benchmark: str = "classification_v1.0"):
    for task in geobench.task_iterator(benchmark_name=benchmark):
        if task.dataset_name == name:
            return task
    raise ValueError(f"Task {name} not found under {benchmark}. Is GEO_BENCH_DIR set?")


def to_hw(arr, h, w):
    if arr.shape[:2] == (h, w):
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False).numpy()[0, 0]


def load_split(task, split, img_size, max_samples=None, seed=0):
    """Load a split into memory, resized to img_size, as float16 (N,C,H,W) + int labels.
    Resizing at load time keeps RAM manageable for large images (pv4ger is 320x320)."""
    ds = task.get_dataset(split=split)
    idx = np.arange(len(ds))
    if max_samples is not None and max_samples < len(ds):
        idx = np.random.default_rng(seed).choice(len(ds), max_samples, replace=False)
    xs, ys = None, np.empty(len(idx), dtype=np.int64)
    for j, i in enumerate(idx):
        s = ds[int(i)]
        bands = [to_hw(np.asarray(b.data, dtype=np.float32), img_size, img_size) for b in s.bands]
        if xs is None:  # preallocate once: no list + np.stack double copy in RAM
            xs = np.empty((len(idx), len(bands), img_size, img_size), dtype=np.float16)
        xs[j] = np.stack(bands)
        ys[j] = int(s.label)
    band_names = [b.band_info.name for b in ds[0].bands]
    return xs, ys, band_names


def band_stats(x, chunk=256):
    """Per-band mean/std computed in chunks: avoids a full float32 copy of the
    dataset in RAM (that copy alone can be several GB)."""
    c = x.shape[1]
    s, ss, n = np.zeros(c), np.zeros(c), 0
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk].astype(np.float64)
        s += b.sum(axis=(0, 2, 3)); ss += (b ** 2).sum(axis=(0, 2, 3))
        n += b.shape[0] * b.shape[2] * b.shape[3]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0)) + 1e-6
    return (mean.reshape(1, c, 1, 1).astype(np.float32),
            std.reshape(1, c, 1, 1).astype(np.float32))


CACHE = Path(os.environ.get("GOTRAIN_CACHE", Path.home() / "gotrain_cache"))


def cached_split(task, split, img_size, max_samples=None, seed=0):
    """Load a full split once, save it as .npy at this img_size, reuse it on every
    later trial (seconds instead of minutes). Subsampling happens after the cache."""
    d = CACHE / f"{task.dataset_name}_{split}_{img_size}"
    if not (d / "y.npy").exists():
        x, y, names = load_split(task, split, img_size)
        tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")  # unique per process
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "x.npy", x)
        (tmp / "bands.json").write_text(json.dumps(names))
        np.save(tmp / "y.npy", y)  # written last: its presence marks a complete cache
        try:
            os.rename(tmp, d)
        except OSError:  # another process finished the same cache first: use theirs
            shutil.rmtree(tmp, ignore_errors=True)
    names = json.loads((d / "bands.json").read_text())
    y = np.load(d / "y.npy")
    if max_samples is not None and max_samples < len(y):
        # memory-map the cache and copy only the chosen rows: RAM ~ subset, not full split
        idx = np.sort(np.random.default_rng(seed).choice(len(y), max_samples, replace=False))
        x = np.ascontiguousarray(np.load(d / "x.npy", mmap_mode="r")[idx])
        return x, y[idx], names
    return np.load(d / "x.npy"), y, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="m-eurosat")
    p.add_argument("--backbone", default="resnet18", help="any timm model, e.g. resnet50, convnext_tiny")
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--time_budget_min", type=float, default=10.0, help="hard wall-clock cap")
    p.add_argument("--patience", type=int, default=5, help="early stop on val accuracy")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_train", type=int, default=None, help="subsample train for fast search")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    t0 = time.time()

    task = get_task(args.task)
    n_classes = task.label_type.n_classes
    xtr, ytr, band_names = cached_split(task, "train", args.img_size, args.max_train, args.seed)
    xva, yva, _ = cached_split(task, "valid", args.img_size)
    load_seconds = round(time.time() - t0, 1)
    t0 = time.time()  # the time budget counts training only, not data loading
    print(f"train {xtr.shape}, val {xva.shape}, {n_classes} classes, bands={band_names} "
          f"(load {load_seconds}s)")
    print(f"train class counts: {np.bincount(ytr, minlength=n_classes).tolist()}")

    # Per-band normalisation from TRAIN stats only
    mean, std = band_stats(xtr)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    xtr, xva = torch.from_numpy(xtr), torch.from_numpy(xva)
    ytr, yva = torch.from_numpy(ytr), torch.from_numpy(yva)

    # timm adapts the first conv to any number of input bands
    model = timm.create_model(args.backbone, pretrained=bool(args.pretrained),
                              in_chans=xtr.shape[1], num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(xtr) + args.batch_size - 1) // args.batch_size)
    budget_s = args.time_budget_min * 60

    def progress(step):
        """Training progress in [0, 1]: whichever runs out first, epochs or time."""
        return min(1.0, max(step / steps, (time.time() - t0) / budget_s))

    def lr_at(p):
        """5% linear warmup, then cosine decay to zero, so the schedule always
        finishes inside the time budget (the low-LR phase is never cut off)."""
        if p < 0.05:
            return args.lr * p / 0.05
        return args.lr * 0.5 * (1 + math.cos(math.pi * (p - 0.05) / 0.95))

    global_step, budget_done = 0, False
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def prep(x, train):
        x = (x.to(device, non_blocking=True).float() - mean_t) / std_t
        if train:  # GEO-Bench protocol: flips + 90-degree rotations
            if torch.rand(1) < 0.5: x = x.flip(-1)
            if torch.rand(1) < 0.5: x = x.flip(-2)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
        return x

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = 0
        for i in range(0, len(xva), 256):
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(prep(xva[i:i + 256], False))
            correct += (logits.argmax(1).cpu() == yva[i:i + 256]).sum().item()
        return correct / len(xva)

    history, best, best_state, bad_epochs, stop_reason = [], -1.0, None, 0, "max_epochs"
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(xtr), args.batch_size):
            p = progress(global_step)
            if p >= 1.0:
                budget_done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(p)
            idx = perm[i:i + args.batch_size]
            x, y = prep(xtr[idx], True), ytr[idx].to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/inf: diverged (try a lower lr)")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            total += loss.item() * len(idx)

        val_acc = evaluate()
        history.append({"epoch": epoch + 1, "train_loss": total / len(xtr), "val_acc": val_acc})
        print(f"epoch {epoch + 1:>2}  loss {total / len(xtr):.4f}  val_acc {val_acc:.4f}")

        if val_acc > best:  # keep the best-on-validation epoch, as GEO-Bench does
            best, bad_epochs = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        # early-stop only in the second half: with cosine decay, the best
        # epochs usually come at the end, once the learning rate is low
        if bad_epochs >= args.patience and progress(global_step) > 0.5:
            stop_reason = "early_stopping"; break
        if budget_done or progress(global_step) >= 1.0:
            stop_reason = "time_budget" if global_step < steps else "max_epochs"; break

    out = Path(args.out)
    if out.name == "runs":  # default: one timestamped folder per run
        out = out / f"{args.task}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "mean": mean, "std": std, "n_classes": n_classes,
                "in_chans": xtr.shape[1], "config": vars(args)}, out / "model.pt")
    best_row = max(history, key=lambda h: h["val_acc"])
    result = {"config": vars(args), "best_val_acc": best_row["val_acc"],
              "best_epoch": best_row["epoch"], "stop_reason": stop_reason,
              "minutes": round((time.time() - t0) / 60, 2), "load_seconds": load_seconds,
              "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
