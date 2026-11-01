"""
Choose each task's FINAL model using the VALIDATION split only (no test data is touched).

For every task it takes the top 3 trials by validation score and tries:
    1, 2 or 3 models averaged  x  with / without test-time augmentation (TTA).
A more complex option is adopted only if it beats the current choice by MIN_GAIN, the same
noise-aware rule the agent uses. The choice is written to final_models.json, which
final_test.py then scores on test, once.

Usage:
    python finalize.py                 # all tasks in registry.json
    python finalize.py m-eurosat       # one task
"""
import json
import sys
from pathlib import Path

import numpy as np

from gotrain_tools import REGISTRY, read_log
from predict import load_data, load_model, predict, resize_probs, score, task_kind

ROOT = Path(__file__).parent
FINAL = ROOT / "final_models.json"
TOP_K = 3
MIN_GAIN = 0.002  # 0.2 points: below this, keep the simpler option


def candidates(task: str) -> list[dict]:
    rows = [r for r in read_log(task) if r["status"] == "ok" and r["val_score"]]
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: float(r["val_score"]), reverse=True):
        path = Path(r["run_dir"]) / "model.pt"
        if r["trial_id"] in seen or not path.exists():
            continue
        seen.add(r["trial_id"])
        cfg = json.loads(r["config"])
        default_img = 128 if task_kind(task) == "classification" else 256
        out.append({"trial_id": r["trial_id"], "model_path": str(path),
                    "img_size": int(cfg.get("img_size", default_img)),
                    "val_score_single": float(r["val_score"])})
        if len(out) == TOP_K:
            break
    return out


def finalize(task: str) -> dict:
    members = candidates(task)
    seg = task_kind(task) == "segmentation"
    ref = members[0]["img_size"]
    _, y = load_data(task, "valid", ref)

    # predictions of each member, without and with TTA (computed once, then combined)
    preds = {False: [], True: []}
    for m in members:
        x, _ = load_data(task, "valid", m["img_size"])
        model, ckpt = load_model(task, m["model_path"])
        for tta in (False, True):
            p = predict(task, model, ckpt, x, tta)
            preds[tta].append(resize_probs(p, ref) if seg else p)
        del model

    results = []
    for k in range(1, len(members) + 1):
        for tta in (False, True):
            s = score(task, np.mean(preds[tta][:k], axis=0), y)
            results.append({"n_models": k, "tta": tta, **s})
            fg = f"  fg IoU {s['iou_fg']:.4f}" if seg else ""
            print(f"   {k} model(s), TTA {'on ' if tta else 'off'}: val {s['score']:.4f}{fg}")

    chosen = results[0]  # simplest: best single model, no TTA
    for r in results[1:]:  # ordered from simple to complex
        if r["score"] > chosen["score"] + MIN_GAIN:
            chosen = r
    print(f"   -> chosen: {chosen['n_models']} model(s), TTA {'on' if chosen['tta'] else 'off'}, "
          f"val {chosen['score']:.4f} (single best was {results[0]['score']:.4f})")
    return {"members": members[:chosen["n_models"]], "tta": chosen["tta"],
            "metric": chosen["metric"], "val_final": chosen["score"],
            "val_single_best": results[0]["score"], "all_options": results}


if __name__ == "__main__":
    tasks = sys.argv[1:] or list(json.loads(REGISTRY.read_text()))
    final = json.loads(FINAL.read_text()) if FINAL.exists() else {}
    for t in tasks:
        print(f"=== {t} ===", flush=True)
        final[t] = finalize(t)
        FINAL.write_text(json.dumps(final, indent=2))  # save after each task
    print(f"\nSaved choices to {FINAL.name}. Nothing was evaluated on test.")
