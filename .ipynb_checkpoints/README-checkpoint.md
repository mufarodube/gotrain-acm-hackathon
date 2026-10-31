# GoTrain: an AI agent that trains geospatial models on its own

Built in 48 hours for the **AUB ACM × RASID hackathon** (31 Oct – 1 Nov 2026).

Give GoTrain a GEO-Bench dataset and it explores the data, chooses a model and training
recipe, trains on a GPU, scores itself on the validation split, and iterates, with no
human in the loop. Every decision is written down as a hypothesis, and every trial is
logged so any score can be traced back to the run that produced it.

The agent runs a ReAct loop with **Claude Sonnet on Amazon Bedrock** through the
**Strands Agents** SDK, on a single **Tesla T4** SageMaker instance.

## Results

| Task | Type | Metric | Hand-written baseline (val) | GoTrain best (val) | Test (once) |
|---|---|---|---|---|---|
| m-eurosat | Classification, 13 bands | Accuracy | 0.948 | 0.985 | _pending_ |
| m-pv4ger | Classification, RGB | Accuracy | – | _running_ | _pending_ |
| m-pv4ger-seg | Segmentation, RGB | mIoU | 0.954 | _running_ | _pending_ |
| m-nz-cattle | Segmentation, RGB | mIoU | – | _running_ | _pending_ |

On m-eurosat the agent also ran the winning configuration with three seeds
(0.981 / 0.985 / 0.976) and concluded in its own report that the realistic level is
about 0.98 and the 0.985 is partly seed luck.

## How it works

```
                 program.md (system prompt: rules, workflow, prior knowledge)
                                     │
 ┌───────────── Strands agent (Claude Sonnet, Bedrock) ─────────────┐
 │  inspect_dataset ──► run_experiment ──► get_history ──► report  │
 └──────────────────────────┬───────────────────────────────────────┘
                            │  one trial at a time (file lock)
            baseline_classification.py / baseline_segmentation.py
                 (train split → validation split, time-budgeted)
                            │
        experiments.csv (audit log) · registry.json (best model per task)
                            │
          final_test.py (run by a human, once per task, locked)
```

### The agent's tools
- **`inspect_dataset`**: split sizes, bands, per-band statistics, class balance or
  foreground-pixel share. The agent reads this before choosing anything.
- **`run_experiment(task, config, hypothesis)`**: trains one model and returns its
  validation score and learning curve. The config is validated against an allowed set of
  keys, and the hypothesis is written to the log.
- **`get_history`**: every past trial on the task, so the agent reasons over results
  rather than guessing.

### Design choices
- **Constrained search, not free-form code.** The agent edits a validated config, not
  raw training code. Trials rarely crash and every result stays comparable.
- **GEO-Bench protocol.** At most **16 trials per task** (the paper's tuning budget),
  augmentations limited to flips and 90° rotations, early stopping on validation, and the
  best validation epoch saved.
- **Fixed time budget per trial**, as in karpathy/autoresearch. The learning-rate schedule
  (warmup, then cosine decay) is **time-aware**: it always completes inside the budget, so
  the low-learning-rate finish is never cut off. On m-pv4ger-seg this alone raised
  validation mIoU from 0.949 to 0.954.
- **Noise-aware keep rule.** A trial becomes the new best only if it beats the current
  best by at least 0.3 points, and the agent confirms the winner with a second seed.
- **Guardrails in code, not just in the prompt.** Trials are serialised with a file lock
  (the agent sometimes calls tools in parallel), RAM use is estimated before launch and
  oversized configs are refused, out-of-memory kills are reported in plain language, and
  diverging losses fail loudly.
- **Test split touched once.** Only `final_test.py` reads it. It evaluates the model in the
  registry and refuses to run twice on the same task.

## Lessons from building it
The first agent run surfaced three real infrastructure bugs, and the agent worked around
each one before we fixed it:
1. **Parallel tool calls** made two trials race on the same cache folder. The agent spotted
   the crashes and switched to sequential runs; we added a lock.
2. **Out-of-memory kills** at 224 px showed up as silent crashes. The agent guessed "cache
   corruption" and moved to 160 px; we cut peak memory, report the real cause, and refuse
   configs that will not fit.
3. **A warning from geo-bench** (m-eurosat's band names are mislabelled, issue #28) buried the
   real error messages; we filter it out of the logs.

Run 1, with those bugs, is kept in `archive/run1/` as evidence.

## Run it yourself

```bash
pip install geobench strands-agents huggingface_hub timm segmentation-models-pytorch
export GEO_BENCH_DIR=~/geobench_data
python download_data.py m-eurosat m-pv4ger m-pv4ger-seg m-nz-cattle

# one task, interactive
python run_agent.py m-eurosat

# several tasks, detached (survives a closed browser)
nohup python -u run_agent.py m-pv4ger-seg m-nz-cattle m-pv4ger > logs/run_all.log 2>&1 &

# at the very end, once per task
python final_test.py m-eurosat
```

Requires access to `eu.anthropic.claude-sonnet-5-5` on Bedrock (`eu-central-1`).

## Files

| File | Role |
|---|---|
| `program.md` | The agent's system prompt: rules, workflow, prior knowledge, hardware limits |
| `run_agent.py` | Runs one fresh agent per task and saves the full conversation to `logs/` |
| `gotrain_tools.py` | The agent's tools, guardrails, experiment log and model registry |
| `baseline_classification.py` | Training engine for classification (timm backbones) |
| `baseline_segmentation.py` | Training engine for segmentation (segmentation-models-pytorch) |
| `final_test.py` | The only code that touches the test split; one evaluation per task |
| `download_data.py` | Downloads only the chosen GEO-Bench tasks from Hugging Face |
| `experiments.csv` | One row per trial: hypothesis, config, validation score, kept or not |
| `registry.json` | The best model per task, with its config |
| `logs/` | Full agent conversations and final reports |

## Next steps toward production (GoPilot)
- Store datasets and model versions in S3, with the registry as the index.
- Expose GoTrain as an MCP server so GoPilot can call `train(dataset)` directly.
- Let the agent edit training code as a second search level once the config search
  plateaus.
- Ensemble the best seeds for the final model.
- Tree search over configurations (AIDE-style) instead of one trial at a time.

## References
- GEO-Bench: Lacoste et al., *GEO-Bench: Toward Foundation Models for Earth Monitoring*
- karpathy/autoresearch: the fixed-budget keep-or-discard loop
- Autoresearch for solar-panel segmentation: Dice + BCE as the strongest single change
