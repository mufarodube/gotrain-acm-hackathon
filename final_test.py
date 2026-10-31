"""
The ONLY script that touches the test split. Run by a human, once per task,
on the model the agent selected (registry.json). A lock in test_results.json
refuses a second evaluation of the same task.

Usage:
    python final_test.py m-eurosat
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gotrain_tools import REGISTRY, task_kind

ROOT = Path(__file__).parent
RESULTS = ROOT / "test_results.json"


@torch.no_grad()
def evaluate(task: str) -> dict:
    entry = json.loads(REGISTRY.read_text())[task]
    ckpt = torch.load(entry["model_path"], map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.from_numpy(ckpt["mean"]).to(device)
    std = torch.from_numpy(ckpt["std"]).to(device)

    if task_kind(task) == "classification":
        import timm
        import baseline_classification as bc
        x, y, _ = bc.load_split(bc.get_task(task), "test", cfg["img_size"])
        model = timm.create_model(cfg["backbone"], pretrained=False, in_chans=ckpt["in_chans"],
                                  num_classes=ckpt["n_classes"])
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        preds = []
        for i in range(0, len(x), 256):
            xb = (torch.from_numpy(x[i:i + 256]).to(device).float() - mean) / std
            preds.append(model(xb).argmax(1).cpu().numpy())
        acc = float((np.concatenate(preds) == y).mean())
        return {"metric": "accuracy", "test_score": acc, "n_test": len(y)}

    import segmentation_models_pytorch as smp
    import baseline_segmentation as bs
    # Note: masks are evaluated at the training img_size (nearest-neighbour resized).
    x, y, _ = bs.load_split(bs.get_task(task), "test", cfg["img_size"])
    model = getattr(smp, cfg["arch"])(encoder_name=cfg["encoder"], encoder_weights=None,
                                      in_channels=ckpt["in_chans"], classes=1)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    inter, union = np.zeros(2), np.zeros(2)
    for i in range(0, len(x), 16):
        xb = (torch.from_numpy(x[i:i + 16]).to(device).float() - mean) / std
        pred = (model(xb) > 0).squeeze(1).cpu().numpy()
        gt = y[i:i + 16].astype(bool)
        for c, (p, g) in enumerate([(~pred, ~gt), (pred, gt)]):
            inter[c] += (p & g).sum(); union[c] += (p | g).sum()
    iou = inter / np.maximum(union, 1)
    return {"metric": "mIoU", "test_score": float(iou.mean()), "test_iou_fg": float(iou[1]),
            "n_test": len(y)}


if __name__ == "__main__":
    task = sys.argv[1]
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if task in results:
        sys.exit(f"REFUSED: {task} was already evaluated on test at {results[task]['timestamp']}.")
    reg = json.loads(REGISTRY.read_text())
    r = evaluate(task)
    r.update({"timestamp": datetime.now(timezone.utc).isoformat(),
              "trial_id": reg[task]["trial_id"], "val_score": reg[task]["val_score"],
              "config": reg[task]["config"]})
    results[task] = r
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(r, indent=2))
