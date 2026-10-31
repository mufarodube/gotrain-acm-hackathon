"""
Hand-written baseline for GEO-Bench binary segmentation (m-pv4ger-seg, m-nz-cattle).
Becomes the core of the agent's segmentation `run_experiment` tool later.

Usage:
    export GEO_BENCH_DIR=~/geobench_data
    python baseline_segmentation.py --task m-pv4ger-seg --img_size 256 --time_budget_min 10
    python baseline_segmentation.py --task m-nz-cattle  --img_size 384 --time_budget_min 10

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
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import geobench
import warnings

# Known geo-bench issue #28: m-eurosat band NAMES are wrong (pixel data is fine).
# We use all bands by position, so this warning only buries real errors.
warnings.filterwarnings("ignore", message=".*incorrect band names.*")


def get_task(name: str, benchmark: str = "segmentation_v1.0"):
    for task in geobench.task_iterator(benchmark_name=benchmark):
        if task.dataset_name == name:
            return task
    raise ValueError(f"Task {name} not found under {benchmark}. Is GEO_BENCH_DIR set?")


def to_hw(arr, h, w, mode):
    """Resize a 2D array to (h, w) if needed."""
    if arr.shape[:2] == (h, w):
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(t, size=(h, w), mode=mode, **kw).numpy()[0, 0]


MAX_BAND_CHANNELS = 32  # skip hyperspectral cubes (e.g. m-NeonTree's 369-channel "Neon" band)


def sample_channels(sample):
    """Split a geobench sample into 2D channels. Bands may be (H, W) or (H, W, C).
    Bands with more than MAX_BAND_CHANNELS channels are skipped: hundreds of inputs on a
    few hundred images overfit, and they would multiply RAM use."""
    chans, names = [], []
    for b in sample.bands:
        a = np.asarray(b.data, dtype=np.float32)
        if a.ndim == 2:
            chans.append(a); names.append(b.band_info.name)
        elif a.ndim == 3 and a.shape[-1] <= MAX_BAND_CHANNELS:
            for k in range(a.shape[-1]):
                chans.append(a[..., k])
                names.append(b.band_info.name if a.shape[-1] == 1 else f"{b.band_info.name}_{k}")
    return chans, names


def load_split(task, split, img_size, max_samples=None):
    """Load a split into memory, resized to img_size. Images float16 (N,C,H,W), masks uint8 (N,H,W)."""
    ds = task.get_dataset(split=split)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    xs, ys = None, np.empty((n, img_size, img_size), dtype=np.uint8)
    for i in range(n):
        s = ds[i]
        chans, _ = sample_channels(s)
        bands = [to_hw(c, img_size, img_size, "bilinear") for c in chans]
        mask = s.label.data if hasattr(s.label, "data") else s.label
        mask = to_hw(np.asarray(mask), img_size, img_size, "nearest")
        if xs is None:  # preallocate once: no list + np.stack double copy in RAM
            xs = np.empty((n, len(bands), img_size, img_size), dtype=np.float16)
        xs[i] = np.stack(bands)
        ys[i] = (mask > 0)  # binary: foreground vs background
    band_names = sample_channels(ds[0])[1]
    skipped = [b.band_info.name for b in ds[0].bands if np.asarray(b.data).ndim == 3
               and np.asarray(b.data).shape[-1] > MAX_BAND_CHANNELS]
    if skipped:
        print(f"note: skipped high-dimensional bands {skipped} (> {MAX_BAND_CHANNELS} channels)")
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
    p.add_argument("--task", default="m-pv4ger-seg")
    p.add_argument("--arch", default="Unet", help="Unet, UnetPlusPlus, DeepLabV3Plus, FPN ...")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--time_budget_min", type=float, default=10.0, help="hard wall-clock cap")
    p.add_argument("--patience", type=int, default=6, help="early stop on val mIoU")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dice_weight", type=float, default=1.0, help="loss = BCE + w * Dice")
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    task = get_task(args.task)
    xtr, ytr, band_names = cached_split(task, "train", args.img_size, args.max_train, args.seed)
    xva, yva, _ = cached_split(task, "valid", args.img_size)
    load_seconds = round(time.time() - t0, 1)
    t0 = time.time()  # the time budget counts training only, not data loading
    print(f"train {xtr.shape}, val {xva.shape}, bands={band_names}, "
          f"foreground pixels {ytr.mean() * 100:.2f}%  (load {load_seconds}s)")

    mean, std = band_stats(xtr)
    xtr, xva = torch.from_numpy(xtr), torch.from_numpy(xva)
    ytr, yva = torch.from_numpy(ytr), torch.from_numpy(yva)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    model = getattr(smp, args.arch)(encoder_name=args.encoder, encoder_weights="imagenet",
                                    in_channels=xtr.shape[1], classes=1).to(device)
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
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def prep(x, y, train):
        x = (x.to(device).float() - mean_t) / std_t
        y = y.to(device).float().unsqueeze(1)
        if train:  # GEO-Bench protocol: flips + 90-degree rotations, applied to image AND mask
            if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        inter = np.zeros(2); union = np.zeros(2)
        for i in range(0, len(xva), 32):
            x, y = prep(xva[i:i + 32], yva[i:i + 32], False)
            with torch.autocast(device, enabled=device == "cuda"):
                pred = (model(x) > 0).float()
            for c, (pc, yc) in enumerate([(1 - pred, 1 - y), (pred, y)]):  # background, foreground
                inter[c] += (pc * yc).sum().item()
                union[c] += ((pc + yc) > 0).sum().item()
        iou = inter / np.maximum(union, 1)
        return {"miou": float(iou.mean()), "iou_fg": float(iou[1]), "iou_bg": float(iou[0])}

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
            x, y = prep(xtr[idx], ytr[idx], True)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + args.dice_weight * dice(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/inf: diverged (try a lower lr)")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            total += loss.item() * len(idx)

        m = evaluate()
        history.append({"epoch": epoch + 1, "train_loss": total / len(xtr), **m})
        print(f"epoch {epoch + 1:>2}  loss {total / len(xtr):.4f}  "
              f"val mIoU {m['miou']:.4f}  fg IoU {m['iou_fg']:.4f}")

        if m["miou"] > best:
            best, bad_epochs = m["miou"], 0
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
    torch.save({"model": best_state, "mean": mean, "std": std, "in_chans": xtr.shape[1],
                "config": vars(args)}, out / "model.pt")
    best_row = max(history, key=lambda h: h["miou"])
    result = {"config": vars(args), "best_val_miou": best_row["miou"],
              "best_val_iou_fg": best_row["iou_fg"], "best_epoch": best_row["epoch"],
              "stop_reason": stop_reason, "minutes": round((time.time() - t0) / 60, 2),
              "load_seconds": load_seconds, "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
