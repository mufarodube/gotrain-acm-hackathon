"""
Run GoTrain on one or more tasks, fully autonomously.

Usage (on the SageMaker instance):
    export GEO_BENCH_DIR=~/geobench_data
    python run_agent.py m-eurosat
    nohup python run_agent.py m-eurosat m-pv4ger m-pv4ger-seg m-nz-cattle > logs/overnight.log 2>&1 &

Each task gets a fresh agent (clean context). The full conversation, including
every tool call and result, is saved to logs/<task>_<time>.json for the judges.
"""
import json
import sys
import time
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

from gotrain_tools import ALL_TOOLS, MAX_TRIALS_PER_TASK

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

MODEL = BedrockModel(model_id="eu.anthropic.claude-sonnet-5-5", region_name="eu-central-1")
SYSTEM_PROMPT = (ROOT / "program.md").read_text()


def run_task(task: str) -> None:
    print(f"\n{'=' * 70}\nGoTrain: {task}\n{'=' * 70}")
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT, tools=ALL_TOOLS)
    t0 = time.time()
    try:
        result = agent(
            f"Your dataset is '{task}'. You have {MAX_TRIALS_PER_TASK} training trials. "
            f"Explore the data, then run experiments to maximise the validation score. "
            f"Finish with your report."
        )
        report = str(result)
    except Exception as e:  # never let one task kill the overnight run
        report = f"AGENT ERROR: {e!r}"
        print(report)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (LOGS / f"{task}_{stamp}.json").write_text(
        json.dumps({"task": task, "minutes": round((time.time() - t0) / 60, 1),
                    "report": report, "messages": agent.messages}, indent=2, default=str))
    (LOGS / f"{task}_{stamp}_report.md").write_text(report)
    print(f"\n=== {task} finished in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    tasks = sys.argv[1:] or ["m-eurosat"]
    for t in tasks:
        run_task(t)
