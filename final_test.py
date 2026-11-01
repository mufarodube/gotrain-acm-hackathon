"""
The ONLY script that touches the test split. Run by a human, once per task.

It scores the final model chosen on validation by finalize.py (final_models.json: an
ensemble and/or test-time augmentation), or, if finalize.py was not run for a task, the
single best model in registry.json. A lock in test_results.json refuses a second
evaluation of the same task.

Usage:
    python final_test.py m-eurosat
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gotrain_tools import REGISTRY, task_kind
from predict import ensemble_probs, score

ROOT = Path(__file__).parent
RESULTS = ROOT / "test_results.json"
FINAL = ROOT / "final_models.json"


def final_choice(task: str) -> dict:
    final = json.loads(FINAL.read_text()) if FINAL.exists() else {}
    if task in final:
        return final[task]
    reg = json.loads(REGISTRY.read_text())[task]  # fallback: single registry model, no TTA
    default_img = 128 if task_kind(task) == "classification" else 256
    return {"members": [{"trial_id": reg["trial_id"], "model_path": reg["model_path"],
                         "img_size": int(reg["config"].get("img_size", default_img))}],
            "tta": False, "val_final": reg["val_score"]}


if __name__ == "__main__":
    task = sys.argv[1]
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if task in results:
        sys.exit(f"REFUSED: {task} was already evaluated on test at {results[task]['timestamp']}.")
    choice = final_choice(task)
    probs, y = ensemble_probs(task, choice["members"], "test", choice["tta"])
    r = score(task, probs, y)
    r.update({"test_score": r.pop("score"), "n_test": int(len(y)),
              "members": [m["trial_id"] for m in choice["members"]], "tta": choice["tta"],
              "val_score": choice["val_final"],
              "timestamp": datetime.now(timezone.utc).isoformat()})
    if task_kind(task) == "segmentation":
        r["note"] = "masks evaluated at the model's img_size (nearest-neighbour resized)"
    results[task] = r
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(r, indent=2))
