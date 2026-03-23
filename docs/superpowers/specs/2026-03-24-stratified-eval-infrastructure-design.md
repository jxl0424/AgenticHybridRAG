# Stratified Eval Infrastructure Design

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Evaluation harness only — no changes to retrieval pipeline

---

## Problem

The current eval harness has three compounding flaws that make results unreliable:

1. **No stratification** — `load_qa_pairs(max_pairs=N)` always pulls the first N rows of `arxiv_ai/qa.parquet`, which happen to be exclusively `counterfactual` questions. The full dataset has 11,497 pairs across 6 question types; the harness has never evaluated on `single_hop`, `multi_hop`, `open_ended`, or `single_hop_w_conditions`.

2. **One-size-fits-all metrics** — `answer_correctness` uses an LLM judge for all types. `single_hop` and `multi_hop` answers are short entity phrases (median 2–3 tokens); LLM comparison against these produces garbage scores.

3. **No variance measurement** — every run uses the same 10 questions. A single eval cannot distinguish real improvement from a lucky question draw.

---

## Design

### 1. Stratified Random Sampling

**Where:** `src/ingestion/local_parquet_loader.py` → `load_local_qa_pairs()`

**New parameters:**
- `seed: int | None` — random seed for reproducibility. If `None`, a random seed is chosen and printed to stdout. Seed is always saved in the output JSON.
- `k_per_type: int` — number of questions sampled per `question_type` (default: 5). With 6 types, `k_per_type=5` produces 30 questions per run.
- `stratify: bool` — when `True` (default), sample `k_per_type` per type. When `False`, random sample `k_per_type * 6` from the full pool.

**Behaviour:** Load all three domain parquet files, concatenate, group by `question_type`, sample `k_per_type` from each group using the given seed, shuffle the combined result, return as a list of dicts with keys: `question`, `answer` (renamed to `ground_truth_answer`), `question_type`, `domain`.

`--max-pairs` CLI arg is retired. It is replaced by `--k-per-type`.

---

### 2. Type-Appropriate Metrics

**Where:** `tests/evaluation/metrics.py`

#### 2a. Normalization

A shared `normalize(text: str) -> str` helper applied before all extractive metrics:
1. Lowercase
2. Remove punctuation
3. Remove articles: `a`, `an`, `the`
4. Normalize whitespace (collapse multiple spaces, strip)

#### 2b. Exact Match (EM)

```
EM = 1.0 if normalize(prediction) == normalize(ground_truth) else 0.0
```

Binary. No partial credit. No fuzzy logic. Fully reproducible.

**New method:** `calculate_exact_match(prediction, ground_truth) -> float`

#### 2c. Token F1

```
tokens_pred = set(normalize(prediction).split())
tokens_gt   = set(normalize(ground_truth).split())
precision   = |tokens_pred ∩ tokens_gt| / |tokens_pred|
recall      = |tokens_pred ∩ tokens_gt| / |tokens_gt|
f1          = 2 * precision * recall / (precision + recall)  # 0.0 if both empty
```

**New method:** `calculate_token_f1(prediction, ground_truth) -> float`

#### 2d. Structured LLM Judge

Replaces the current free-form prompt for `open_ended` and `counterfactual` types.

**Changes to existing LLM judge calls:**
- Temperature = 0 (already set; verified enforced)
- Fixed rubric anchors in prompt:
  - 1.0 = fully correct
  - 0.7–0.9 = mostly correct, minor omissions
  - 0.3–0.6 = partially correct
  - 0.0–0.2 = wrong or unjustified refusal
- Required output format: `Score: <float>\nJustification: <one sentence>`
- Parse score with regex; store justification in per-question JSON

**Counterfactual-specific clause** added to the judge prompt:
> "If the model correctly identifies that a premise is false or cannot be verified from the provided context, this is a valid and high-scoring response. Score it highly."

#### 2e. Metric dispatch by question_type

| question_type | EM | Token F1 | LLM judge |
|---|---|---|---|
| single_hop | yes | yes | no |
| single_hop_w_conditions | yes | yes | no |
| multi_hop | yes | yes | no |
| multi_hop_difficult | yes | yes | escalate if `len > 15 tokens AND token_f1 == 0` |
| open_ended | no | no | yes |
| counterfactual | no | no | yes |

**`final_answer_correctness`** — unified headline metric:
- Extractive types: = `token_f1`
- Generative types (or escalated multi_hop_difficult): = LLM judge score

This is the primary metric for tracking system-level progress over time.

---

### 3. Multi-Run Support

**Where:** `tests/evaluation/hybridrag_eval.py`

**New CLI args:**

| arg | default | description |
|---|---|---|
| `--seed INT` | random | Controls sampling. Always printed and saved in JSON. |
| `--runs N` | 1 | Run N trials with seeds `[seed, seed+1, ..., seed+N-1]` |
| `--k-per-type INT` | 5 | Questions per type per run |

**Run loop:** When `--runs N > 1`, execute `N` independent eval runs. Each run uses a fresh sample drawn with `seed + run_index`. Per-run results are saved to `{output}_run{i}.json`. After all runs complete, a summary is written to `{output}_summary.json`.

---

### 4. Output Schema

#### Per-question entry (in `run{i}.json`)

```json
{
  "question": "...",
  "question_type": "single_hop",
  "prediction": "...",
  "ground_truth": "...",
  "seed": 42,
  "run": 1,
  "metrics": {
    "exact_match": 1.0,
    "token_f1": 1.0,
    "answer_correctness_llm": null,
    "lm_judge_justification": null,
    "final_answer_correctness": 1.0,
    "context_recall": 0.75,
    "faithfulness": 0.85
  },
  "num_contexts": 5,
  "trace": { "...": "..." }
}
```

#### Summary file (`summary.json`)

```json
{
  "config": { "seed": 42, "runs": 5, "k_per_type": 5, "modes": ["hybrid"] },
  "per_metric": {
    "final_answer_correctness": { "mean": 0.74, "std": 0.03, "median": 0.75, "min": 0.69, "max": 0.79 },
    "context_recall":           { "mean": 0.51, "std": 0.04, "median": 0.51, "min": 0.46, "max": 0.57 },
    "faithfulness":             { "mean": 0.81, "std": 0.03, "median": 0.81, "min": 0.76, "max": 0.86 }
  },
  "per_type_per_metric": {
    "single_hop":    { "final_answer_correctness": { "mean": 0.81, "std": 0.03 }, "..." : "..." },
    "counterfactual": { "final_answer_correctness": { "mean": 0.69, "std": 0.05 }, "...": "..." }
  }
}
```

**Overall** = macro average across question types (equal weight per type, consistent with balanced sampling).

---

### 5. Display

Two print methods updated in `HybridRAGEvaluator`:

**`print_type_breakdown(aggregated)`** — primary view, shows all metrics split by type with clear section headers separating extractive metrics (EM, Token F1) from LLM judge metrics (answer_correctness_llm) and the unified metric (final_answer_correctness).

**`print_multirun_summary(summary)`** — shown when `--runs > 1`. Displays mean, std, median, min, max for every scalar metric. Includes per-type std for `final_answer_correctness`.

---

## Files Changed

| File | Change |
|---|---|
| `src/ingestion/local_parquet_loader.py` | Add `seed`, `k_per_type`, `stratify` params to `load_local_qa_pairs()` |
| `tests/evaluation/metrics.py` | Add `normalize()`, `calculate_exact_match()`, `calculate_token_f1()`; update LLM judge prompt (structured output, rubric anchors, counterfactual clause) |
| `tests/evaluation/hybridrag_eval.py` | Add `--seed`, `--runs`, `--k-per-type` CLI args; retire `--max-pairs`; update `_compute_metrics()` to dispatch on `question_type`; add multi-run loop; add `print_type_breakdown()` and `print_multirun_summary()`; update output schema |

## Out of Scope

- Changes to the retrieval pipeline (`hybridrag_pipeline.py`, `hybrid_retriever.py`, `graph_retriever.py`)
- Multi-hop retrieval improvements (planned for next phase)
- New ingestion or embedding changes
