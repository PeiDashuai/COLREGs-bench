# COLREGs Structured Reasoning: Code

This repository contains the public implementation used to read and validate
COLREGs-Bench, parse and score structured predictions, run the Symbolic COLREGs
Rule Engine, project outputs into the common semantic space, and reproduce the
training/inference protocol components reported in the paper.

The benchmark is distributed separately as the `COLREGs-Bench-final` release
asset because its 3,400 PNG files are not suitable for ordinary Git history.
Place the extracted benchmark next to this repository, or pass its path directly
to each command.

## Install

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

For model training or local Gemma inference:

```bash
python -m pip install -e ".[training]"
```

For GPT-series API inference:

```bash
python -m pip install -e ".[api]"
```

## Verify and inspect the benchmark

```bash
python scripts/verify_release.py ../COLREGs-Bench-final
python scripts/inspect_benchmark.py ../COLREGs-Bench-final \
  --split test --view expert_corrected_core3 --index 0
```

The release exposes three evaluation views with different scientific roles:

- `full_structured`: complete structured references used for six-field
  diagnostic evaluation and for deriving training targets;
- `expert_corrected_core3`: the final expert-corrected test reference for
  triggered rules, allowed maneuvers, and forbidden maneuvers;
- `common_semantic`: the 39-concept ontology and 183 reviewed correspondences
  used to compare the rule engine and learned systems in one scoring space.

These views must be scored separately. Results from different views are not
directly comparable.

## Score predictions

Prediction rows require a unique `sample_id`. The predicted object may be at the
row root or under `prediction`.

```bash
python scripts/score_predictions.py \
  ../COLREGs-Bench-final predictions.jsonl score_out \
  --view expert_corrected_core3 --split test --run-id my_run
```

## Run the symbolic baseline

```bash
python scripts/run_symbolic_baseline.py \
  ../COLREGs-Bench-final symbolic_predictions.jsonl
```

Project the benchmark reference and symbolic output to common semantics, then
score them:

```bash
python scripts/project_common_semantics.py \
  --source-space benchmark \
  --input ../COLREGs-Bench-final/evaluation/expert_corrected_core3/test.jsonl \
  --mapping ../COLREGs-Bench-final/evaluation/common_semantic/benchmark_mapping.json \
  --ontology ../COLREGs-Bench-final/evaluation/common_semantic/ontology.json \
  --schema schemas/common_semantic_projection.schema.json \
  --output gold_common.jsonl

python scripts/project_common_semantics.py \
  --source-space symbolic_engine \
  --input symbolic_predictions.jsonl \
  --mapping ../COLREGs-Bench-final/evaluation/common_semantic/symbolic_engine_mapping.json \
  --ontology ../COLREGs-Bench-final/evaluation/common_semantic/ontology.json \
  --schema schemas/common_semantic_projection.schema.json \
  --output symbolic_common.jsonl

python scripts/score_common_semantics.py \
  --gold gold_common.jsonl --prediction symbolic_common.jsonl \
  --output-dir common_score --run-id symbolic_rule_engine
```

## API inference and credential safety

No credential is stored in this repository. `run_api_inference.py` reads the
credential only from an environment variable. For an OpenAI-compatible endpoint:

Set `OPENAI_API_KEY` in the process environment and, for a compatible third-party
endpoint, set `OPENAI_BASE_URL`. Then run:

```bash
python scripts/run_api_inference.py \
  ../COLREGs-Bench-final api_predictions.jsonl \
  --model provider/model-id --scope full --modality text
```

Do not commit `.env` files, prediction artifacts, adapters, model weights, or API
responses. The included `.gitignore` blocks the common local locations.

## Implementation map

- `src/colregs_framework/benchmark.py`: release reader and integrity validator;
- `benchmark_generation/`: final scene generation, labeling, acceptance,
  rendering, split construction, and benchmark QA code;
- `parsing.py`, `scoring.py`: strict structured-output parsing and deterministic scoring;
- `datasets.py`, `training.py`, `trainer_backend.py`: shared training adapters and Gemma LoRA backend;
- `inference.py`, `inference_backend.py`: resumable local/API inference components;
- `symbolic_*.py`, `src/colreg_kernel`, `src/colreg_reasoner`: symbolic rule execution and multi-target resolution;
- `semantic_projection.py`, `symbolic_scoring.py`: common-semantic comparison;
- `robustness.py`, `robustness_inference.py`: the fixed robustness transformations and inference handling.

`configs/paper_experiment_config.json` records the paper-level settings without
machine-specific paths, credentials, model weights, or internal run artifacts.
