"""
GoTrain tools: what the agent is allowed to do.

The agent can look at data, run training trials, and read its history.
It CANNOT touch the test split: that is done once, by a human, with final_test.py.
Every trial is logged to experiments.csv (the audit trail RASID asks for),
and the best model per task is tracked in registry.json.
"""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from strands import tool

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
LOG_CSV = ROOT / "experiments.csv"
REGISTRY = ROOT / "registry.json"
CACHE = Path(os.environ.get("GOTRAIN_CACHE", Path.home() / "gotrain_cache"))

# Known geo-bench issue #28 (m-eurosat band NAMES are wrong; pixel data is fine). Harmless here.
warnings.filterwarnings("ignore", message=".*incorrect band names.*")

MAX_TRIALS_PER_TASK = 16          # GEO-Bench recommended tuning budget
MAX_TIME_BUDGET_MIN = 15.0        # hard cap on any single trial
MIN_IMPROVEMENT = 0.003           # a change must beat the best by this margin to be "kept"
MAX_DATA_GB = 7.0                 # RAM allowed for one trial's in-memory dataset (instance: ~15 GB)
DEFAULT_IMG = {"classification": 128, "segmentation": 256}

SEG_TASKS = {"m-pv4ger-seg", "m-chesapeake", "m-cashew-plant",
             "m-SA-crop-type", "m-nz-cattle", "m-NeonTree"}

ALLOWED = {
    "classification": {"backbone", "pretrained", "img_size", "epochs", "time_budget_min",
                       "patience", "batch_size", "lr", "weight_decay", "max_train", "seed"},
    "segmentation": {"arch", "encoder", "img_size", "epochs", "time_budget_min", "patience",
                     "batch_size", "lr", "weight_decay", "dice_weight", "max_train", "seed"},
}
SCRIPT = {"classification": "baseline_classification.py",
          "segmentation": "baseline_segmentation.py"}
METRIC = {"classification": "best_val_acc", "segmentation": "best_val_miou"}

CSV_FIELDS = ["trial_id", "task", "timestamp", "status", "hypothesis", "config",
              "val_score", "kept", "best_so_far", "minutes", "run_dir", "error"]


# ---------- helpers ----------

def task_kind(task: str) -> str:
    return "segmentation" if task in SEG_TASKS else "classification"


def read_log(task: str | None = None) -> list[dict]:
    if not LOG_CSV.exists():
        return []
    with LOG_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if task is None or r["task"] == task]


def append_log(row: dict) -> None:
    new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2))


@lru_cache(maxsize=None)
def dataset_shape(task: str) -> tuple[int, int, int]:
    """(n_train, n_val, n_bands) for a task, read once from geobench."""
    import geobench
    bench = "segmentation_v1.0" if task_kind(task) == "segmentation" else "classification_v1.0"
    t = next(x for x in geobench.task_iterator(benchmark_name=bench) if x.dataset_name == task)
    train, val = t.get_dataset(split="train"), t.get_dataset(split="valid")
    channels = 0  # same rule as the training scripts: (H,W) = 1, (H,W,C<=32) = C, bigger = skipped
    for b in train[0].bands:
        shape = np.asarray(b.data).shape
        channels += 1 if len(shape) == 2 else (shape[-1] if shape[-1] <= 32 else 0)
    return len(train), len(val), channels


def memory_check(task: str, kind: str, config: dict) -> str | None:
    """Estimate the RAM a trial needs BEFORE launching it. Returns a refusal
    message if it would not fit (an out-of-memory kill can freeze the instance)."""
    n_tr, n_va, bands = dataset_shape(task)
    img = int(config.get("img_size", DEFAULT_IMG[kind]))
    bytes_per_px = bands * 2 + (1 if kind == "segmentation" else 0)   # float16 bands + uint8 mask
    cached = (CACHE / f"{task}_train_{img}" / "y.npy").exists()
    # building a new cache loads the full split; an existing cache can be subsampled cheaply
    n_used = min(n_tr, int(config["max_train"])) if config.get("max_train") and cached else n_tr
    gb = (n_used + n_va) * img * img * bytes_per_px / 1e9
    if gb <= MAX_DATA_GB:
        return None
    max_img = int((MAX_DATA_GB * 1e9 / ((n_used + n_va) * bytes_per_px)) ** 0.5) // 32 * 32
    return (f"REFUSED (no trial used): img_size {img} needs about {gb:.1f} GB of RAM for "
            f"{n_used + n_va} images x {bands} bands, above the {MAX_DATA_GB:.0f} GB limit. "
            f"Use img_size <= {max_img} for this dataset.")


def _last_progress(log_path: Path) -> str:
    """Latest epoch line from a running trial's log, for the live heartbeat."""
    try:
        lines = [l.strip() for l in log_path.read_text().splitlines() if l.startswith("epoch")]
        return lines[-1] if lines else "loading data / starting"
    except OSError:
        return "starting"


# ---------- tools ----------

@tool
def inspect_dataset(task: str, n_samples: int = 200) -> str:
    """Look at a GEO-Bench dataset before training anything.

    Returns split sizes, image shape, band names, per-band statistics, and the
    label distribution (class counts for classification, foreground-pixel share
    for segmentation). Uses the TRAIN split only.

    Args:
        task: dataset name, e.g. "m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-nz-cattle".
        n_samples: how many training samples to scan for statistics.
    """
    import geobench
    kind = task_kind(task)
    bench = "segmentation_v1.0" if kind == "segmentation" else "classification_v1.0"
    t = next((x for x in geobench.task_iterator(benchmark_name=bench) if x.dataset_name == task), None)
    if t is None:
        return f"ERROR: task {task} not found. Is it downloaded and GEO_BENCH_DIR set?"

    train, val = t.get_dataset(split="train"), t.get_dataset(split="valid")
    s0 = train[0]
    info = {
        "task": task, "kind": kind, "n_train": len(train), "n_val": len(val),
        "bands": [{"name": b.band_info.name, "shape": list(np.asarray(b.data).shape)} for b in s0.bands],
    }
    n = min(n_samples, len(train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), n, replace=False)
    stats = [[] for _ in s0.bands]
    labels = []
    for i in idx:
        s = train[int(i)]
        for j, b in enumerate(s.bands):
            a = np.asarray(b.data, dtype=np.float32)
            stats[j].append((a.mean(), a.std()))
        if kind == "classification":
            labels.append(int(s.label))
        else:
            m = s.label.data if hasattr(s.label, "data") else s.label
            labels.append(float((np.asarray(m) > 0).mean()))
    info["band_stats"] = {b["name"]: {"mean": round(float(np.mean([m for m, _ in st])), 3),
                                      "std": round(float(np.mean([s for _, s in st])), 3)}
                          for b, st in zip(info["bands"], stats)}
    if kind == "classification":
        info["n_classes"] = t.label_type.n_classes
        info["class_counts_in_sample"] = np.bincount(labels, minlength=info["n_classes"]).tolist()
    else:
        info["foreground_pixel_share"] = {"mean": round(float(np.mean(labels)), 4),
                                          "share_of_images_with_any_foreground":
                                              round(float(np.mean(np.array(labels) > 0)), 3)}
    info["allowed_config_keys"] = sorted(ALLOWED[kind])
    info["trials_used"] = len(read_log(task))
    info["trial_budget"] = MAX_TRIALS_PER_TASK
    return json.dumps(info, indent=2)


@tool
def run_experiment(task: str, config: dict, hypothesis: str) -> str:
    """Train one model on the TRAIN split and score it on the VALIDATION split.

    Each call uses one trial from the task's budget (16 per task). The result is
    logged to experiments.csv. The model is marked "kept" only if it beats the
    best validation score so far by a minimum margin; kept models are recorded
    in the model registry.

    Args:
        task: dataset name, e.g. "m-eurosat".
        config: training settings. Only keys listed in inspect_dataset's
            "allowed_config_keys" are accepted. Classification example:
            {"backbone": "resnet50", "lr": 0.0003, "img_size": 128, "epochs": 15}.
            Segmentation example:
            {"arch": "Unet", "encoder": "resnet34", "img_size": 256, "dice_weight": 1.0}.
        hypothesis: one or two sentences: what you changed versus the current best
            and why you expect it to help. This is written to the audit log.
    """
    # Tool calls can arrive in parallel; trials must run one at a time
    # (one GPU, unique trial ids, clean log). This lock queues them.
    with open(ROOT / ".trial.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _run_experiment(task, config, hypothesis)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_experiment(task: str, config: dict, hypothesis: str) -> str:
    kind = task_kind(task)
    used = len(read_log(task))
    if used >= MAX_TRIALS_PER_TASK:
        return (f"REFUSED: trial budget for {task} is used up ({used}/{MAX_TRIALS_PER_TASK}). "
                f"Summarise your findings and stop.")
    bad = set(config) - ALLOWED[kind]
    if bad:
        return f"REFUSED: unknown config keys {sorted(bad)}. Allowed: {sorted(ALLOWED[kind])}"
    config = dict(config)
    if kind == "segmentation" and int(config.get("img_size", 256)) % 32:
        return ("REFUSED: segmentation img_size must be a multiple of 32 "
                "(e.g. 256, 384, 480, 512) because the U-Net halves it 5 times.")
    config["time_budget_min"] = min(float(config.get("time_budget_min", 10)), MAX_TIME_BUDGET_MIN)
    too_big = memory_check(task, kind, config)
    if too_big:
        print(f"\n   {too_big}", flush=True)
        return too_big

    trial_id = f"{task}_t{used + 1:02d}"
    run_dir = RUNS / task / trial_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / SCRIPT[kind]), "--task", task, "--out", str(run_dir)]
    for k, v in config.items():
        cmd += [f"--{k}", str(v)]

    print(f"\n>> {trial_id}  (trial {used + 1}/{MAX_TRIALS_PER_TASK})\n"
          f"   hypothesis: {hypothesis}\n   config: {json.dumps(config)}", flush=True)

    t0 = time.time()
    status, error, score = "ok", "", None
    log_path = run_dir / "stdout.log"
    limit_s = config["time_budget_min"] * 60 + 900  # training budget + data loading slack
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # so epoch lines reach the log live
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, env=env)
        last_beat = 0.0
        while proc.poll() is None:
            time.sleep(2)
            elapsed = time.time() - t0
            if elapsed > limit_s:
                proc.kill(); proc.wait()
                status, error = "timeout", "trial exceeded its time limit and was killed"
                break
            if elapsed - last_beat >= 30:  # heartbeat every 30 s
                last_beat = elapsed
                print(f"   ... {int(elapsed // 60)}m{int(elapsed % 60):02d}s | "
                      f"{_last_progress(log_path)}", flush=True)
    if status == "ok" and proc.returncode != 0:
        status, error = "crashed", log_path.read_text()[-1500:]
        if proc.returncode < 0:  # killed by a signal, e.g. SIGKILL from the out-of-memory killer
            status = "killed"
            error = (f"Process killed by the system (signal {-proc.returncode}), almost certainly "
                     f"OUT OF MEMORY (RAM). Use a smaller img_size, max_train or batch_size. "
                     f"This is not a code bug.\n" + error[-500:])

    metrics = {}
    if status == "ok" and (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
        score = metrics[METRIC[kind]]
    elif status == "ok":
        status, error = "crashed", "no metrics.json produced"

    reg = load_registry()
    best = reg.get(task, {}).get("val_score", -1.0)
    kept = score is not None and score > best + MIN_IMPROVEMENT
    if kept:
        reg[task] = {"trial_id": trial_id, "val_score": score, "metric": METRIC[kind],
                     "config": config, "model_path": str(run_dir / "model.pt"),
                     "updated": datetime.now(timezone.utc).isoformat()}
        save_registry(reg)
        best = score

    append_log({"trial_id": trial_id, "task": task,
                "timestamp": datetime.now(timezone.utc).isoformat(), "status": status,
                "hypothesis": hypothesis, "config": json.dumps(config),
                "val_score": "" if score is None else round(score, 5), "kept": kept,
                "best_so_far": round(best, 5) if best >= 0 else "",
                "minutes": round((time.time() - t0) / 60, 2), "run_dir": str(run_dir),
                "error": error[-300:]})

    score_txt = "n/a" if score is None else f"{score:.4f}"
    best_txt = f"{best:.4f}" if best >= 0 else "n/a"
    print(f"   [{status}] val {score_txt} | {'KEPT, new best' if kept else 'not kept'} | "
          f"best so far {best_txt} | {(time.time() - t0) / 60:.1f} min", flush=True)

    summary = {"trial_id": trial_id, "status": status, "val_score": score, "kept": kept,
               "best_val_score_so_far": best if best >= 0 else None,
               "trials_left": MAX_TRIALS_PER_TASK - used - 1}
    if metrics:
        summary.update({k: metrics[k] for k in ("best_epoch", "stop_reason", "minutes",
                                                 "best_val_iou_fg") if k in metrics})
        summary["val_curve"] = [round(h.get("val_acc", h.get("miou", 0)), 4)
                                for h in metrics.get("history", [])]
    if error:
        summary["error_tail"] = error[-800:]
    return json.dumps(summary, indent=2)


@tool
def get_history(task: str) -> str:
    """Return every trial run so far on a task: config, hypothesis, validation
    score, whether it was kept, and errors. Read this before proposing the next trial.

    Args:
        task: dataset name, e.g. "m-eurosat".
    """
    rows = read_log(task)
    if not rows:
        return f"No trials yet for {task}."
    keep = ["trial_id", "status", "val_score", "kept", "hypothesis", "config", "error"]
    reg = load_registry().get(task)
    return json.dumps({"current_best": reg, "trials": [{k: r[k] for k in keep} for r in rows]},
                      indent=2)


ALL_TOOLS = [inspect_dataset, run_experiment, get_history]
