"""
Download only the GEO-Bench tasks you need from Hugging Face.

Usage (in a JupyterLab terminal):
    pip install geobench strands-agents huggingface_hub timm segmentation-models-pytorch
    python download_data.py m-eurosat
    python download_data.py m-pv4ger m-pv4ger-seg m-nz-cattle

Data lands in ~/geobench_data (override with GEO_BENCH_DIR).
"""
import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "recursix/geo-bench-1.0"
ROOT = Path(os.environ.get("GEO_BENCH_DIR", Path.home() / "geobench_data"))

SEG_TASKS = {"m-pv4ger-seg", "m-chesapeake", "m-cashew-plant",
             "m-SA-crop-type", "m-nz-cattle", "m-NeonTree"}


def download(task: str) -> None:
    benchmark = "segmentation_v1.0" if task in SEG_TASKS else "classification_v1.0"
    target_dir = ROOT / benchmark
    target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / task).exists():
        print(f"[skip] {task} already at {target_dir / task}")
        return

    files = list_repo_files(REPO, repo_type="dataset")
    # Grab the task zip plus any normalizer/metadata files for that benchmark
    wanted = [f for f in files
              if f.startswith(f"{benchmark}/") and
              (f.endswith(f"/{task}.zip") or not f.endswith(".zip"))]
    if not any(f.endswith(f"{task}.zip") for f in wanted):
        sys.exit(f"Could not find {task}.zip in {benchmark}/. Available: "
                 f"{[f for f in files if f.startswith(benchmark)]}")

    for f in wanted:
        print(f"[download] {f}")
        local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=ROOT)
        if local.endswith(".zip"):
            print(f"[unzip] {local} -> {target_dir}")
            with zipfile.ZipFile(local) as z:
                z.extractall(target_dir)
            os.remove(local)  # free disk space

    print(f"[done] {task}: {sorted(p.name for p in (target_dir).iterdir())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python download_data.py <task> [<task> ...]")
    for t in sys.argv[1:]:
        download(t)
    print(f"\nNow run:  export GEO_BENCH_DIR={ROOT}")
