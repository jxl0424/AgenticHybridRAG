# Stratified Eval Infrastructure Design

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Evaluation harness only — no changes to retrieval pipeline

---

## Problem

The current eval harness has three compounding flaws that make results unreliable:

1. **No stratification** — `load_qa_pairs(max_pairs=N)` always pulls the first N rows of `arxiv_ai/qa.parquet`, which happen to be exclusively `counterfactual` questions. The full dataset has 11,497 pairs across 6 question types (exact `question_type` column values): `single_hop`, `single_hop_w_conditions`, `multi_hop`, `multi_hop_difficult`, `open_ended`, `counterfactual`. The harness has never evaluated on `single_hop`, `multi_hop`, `open_ended`, or `single_hop_w_conditions`.

2. **One-size-fits-all metrics** — `answer_correctness` uses an LLM judge for all types. `single_hop` and `multi_hop` answers are short entity phrases (median 2–3 tokens); LLM comparison against these produces garbage scores.

3. **No variance measurement** — every run uses the same questions. A single eval cannot distinguish real improvement from a lucky question draw.

---

## Dataset Notes

- The parquet files (`text_qa/{domain}/qa.parquet`) contain no `ground_truth_context` column. As a result, `hit_rate`, `mrr`, `context_precision`, and `ndcg_at_5` cannot be computed for this benchmark and will always be `N/A`. These metrics are retained in the code but skipped at runtime.
- All three domains are fully ingested. QA pairs span `arxiv_ai` (863), `arxiv_cy` (5,219), `arxiv_qm` (5,415). Sampling is stratified by `question_type`, not domain — domain representation within each run is random and unbalanced. This is noted in the summary config for transparency.

---

## Design

### 1. Stratified Random Sampling

**Where:** `src/ingestion/local_parquet_loader.py` → `load_local_qa_pairs()`

**New parameters:**
- `seed: int | None` — random seed. If `None`, a seed is drawn using `int(np.random.default_rng().integers(0, 1_000_000))`, printed by the eval entry point as `Seed: {seed}` before the first run begins, and saved in every output JSON. Print location: `hybridrag_eval.py` (entry point only, not inside the loader).
- `k_per_type: int` — number of questions sampled per `question_type` (default: 5). With 6 types this yields 30 questions per run. When a type has fewer rows than `k_per_type`, all rows for that type are used and a warning is printed: `WARNING: only {n} rows available for {type}, using all`.
- `stratify: bool` — when `True` (default), sample `k_per_type` per type. When `False`, random sample `k_per_type * 6` from the full pool without type balancing.

**Backward compatibility:** `max_pairs` is removed from `load_local_qa_pairs()`. The `load_qa_pairs()` wrapper in `HybridRAGEvaluator` is updated to accept `seed`, `k_per_type`, and `stratify` and pass them through. The `max_pairs` argument is also removed from the evaluator wrapper.

**RNG:** A single `numpy.random.Generator` (`rng = np.random.default_rng(seed)`) is used for all randomness in the sampling step — both per-group selection and the final shuffle. This eliminates RNG state mixing between pandas/numpy and Python's `random` module.

- Per-group sampling: `group_df.sample(n=k_per_type, random_state=rng)` (pandas accepts a numpy Generator since v1.4)
- Final shuffle: `idx = rng.permutation(len(combined)); combined = [combined[i] for i in idx]`
- `stratify=False` path: `full_df.sample(n=k_per_type * 6, random_state=rng)`

Using the same `rng` instance across all steps ensures full reproducibility from a single seed integer.

**`--output` validation:** The `--output` path must end in `.json`. If it does not, the CLI raises a `ValueError` at startup before any eval runs.

---

### 2. Type-Appropriate Metrics

**Where:** `tests/evaluation/metrics.py`

#### 2a. Normalization

A shared `normalize(text: str) -> str` helper applied before all extractive metrics:
1. Unicode NFKD normalization, encode to ASCII (drops accents, normalizes ligatures)
2. Lowercase
3. Remove punctuation (replace with space)
4. Remove articles: exact token matches of `a`, `an`, `the`
5. Normalize whitespace (collapse multiple spaces, strip)

#### 2b. Exact Match (EM)

```
EM = 1.0 if normalize(prediction) == normalize(ground_truth) else 0.0
```

Binary. No partial credit. No fuzzy logic. Fully reproducible.

**New method:** `calculate_exact_match(prediction: str, ground_truth: str) -> float`

#### 2c. Token F1

Uses **token bags (multisets)** via `collections.Counter`, not sets. This matches the SQuAD reference implementation and correctly handles repeated tokens.

```python
from collections import Counter

pred_tokens = normalize(prediction).split()
gt_tokens   = normalize(ground_truth).split()
pred_counts = Counter(pred_tokens)
gt_counts   = Counter(gt_tokens)
common      = sum((pred_counts & gt_counts).values())  # bag intersection

precision = common / len(pred_tokens) if pred_tokens else 0.0
recall    = common / len(gt_tokens)   if gt_tokens   else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
```

**New method:** `calculate_token_f1(prediction: str, ground_truth: str) -> float`

#### 2d. Structured LLM Judge

Replaces the current free-form prompt for `open_ended` and `counterfactual` types.

**Changes to existing LLM judge calls:**
- Temperature = 0 (already set; confirmed enforced via `LLMClient.generate()`)
- Fixed, non-overlapping rubric anchors:
  - 0.7–1.0 = mostly or fully correct (minor omissions acceptable)
  - 0.3–0.69 = partially correct (key points missing or inaccurate)
  - 0.0–0.29 = wrong, off-topic, or unjustified refusal
- Required output format: `Score: <float>\nJustification: <one sentence>`
- Parse score with `re.search(r"Score:\s*(\d*\.?\d+)")`. Clamp parsed value to `[0.0, 1.0]` via `min(1.0, max(0.0, score))`. Store the extracted justification sentence in `lm_judge_justification`, truncated to 500 characters on success.
- **Parse failure handling:** If the regex does not match, log `WARNING: LLM judge parse failed for question "{question[:60]}..."`, set `answer_correctness_llm` to `null`, set `lm_judge_justification` to the raw LLM response truncated to 200 characters, set `final_answer_correctness` to `null`, and set `final_answer_correctness_source` to `null` for that question. The question still appears in results with `null` metrics — it is not silently dropped.

**Counterfactual-specific clause** added to the judge prompt:
> "If the model correctly identifies that a premise is false or cannot be verified from the provided context, this is a valid and high-scoring response. Score it as 0.7 or higher."

#### 2e. Metric dispatch by question_type

| question_type | EM | Token F1 | LLM judge | escalation rule |
|---|---|---|---|---|
| single_hop | yes | yes | no | — |
| single_hop_w_conditions | yes | yes | no | — |
| multi_hop | yes | yes | no | — |
| multi_hop_difficult | yes | yes | conditional | if `token_f1 == 0.0 AND len(normalize(prediction).split()) > 15` |
| open_ended | no | no | yes | — |
| counterfactual | no | no | yes | — |

**`multi_hop_difficult` escalation rationale:** The escalation signal is the *prediction* length, not the ground truth. A short prediction with F1=0 is simply wrong — the LLM judge would also score it 0. A long prediction with F1=0 indicates the model gave a verbose answer that token matching cannot evaluate; escalating to the LLM judge gives it a fair chance. Ground truth length is a poor signal because >97% of `multi_hop_difficult` ground truths are ≤15 tokens.

**Null population for inapplicable metrics:** For metrics not applicable to a question type, the field is written as `null` in the per-question JSON (not omitted, not `"N/A"`). This keeps the schema uniform across all entries. When escalation fires, `final_answer_correctness_source` is set to `"llm"` in the per-question JSON (vs `"token_f1"` otherwise) so downstream analysis can filter by source.

#### 2f. `final_answer_correctness` — unified headline metric

**Warning: mixed-scale metric.** Extractive types use Token F1 (bounded by token overlap); generative types use LLM judge scores (bounded by model calibration). These are not on the same scale. The "overall" macro average is valid only for tracking relative improvement within this system over time — it must not be used to compare this system to external benchmarks or to interpret absolute performance.

| question_type | value |
|---|---|
| Extractive (single_hop, single_hop_w_conditions, multi_hop) | = `token_f1` |
| multi_hop_difficult (no escalation) | = `token_f1` |
| multi_hop_difficult (escalated) | = LLM judge score |
| open_ended, counterfactual | = LLM judge score |

---

### 3. Multi-Run Support

**Where:** `tests/evaluation/hybridrag_eval.py`

**New CLI args:**

| arg | default | description |
|---|---|---|
| `--seed INT` | random | Controls all sampling. Printed as `Seed: {seed}` before run 1. Saved in every output JSON. |
| `--runs N` | 1 | Run N trials with seeds `[seed, seed+1, ..., seed+N-1]` |
| `--k-per-type INT` | 5 | Questions per type per run (30 total with 6 types) |

`--max-pairs` is removed.

**Run loop:** When `--runs N > 1`, execute N independent eval runs. Output files are named by inserting `_run{i}` before the `.json` extension of the `--output` path: e.g., for `--output tests/results/my_eval.json`, run files are `my_eval_run1.json`, `my_eval_run2.json`. The summary is named by inserting `_summary` before `.json`: `my_eval_summary.json`. This derivation applies to any `--output` value.

---

### 4. Output Schema

#### Per-question entry (in `_run{i}.json`)

```json
{
  "question": "...",
  "question_type": "single_hop",
  "domain": "arxiv_ai",
  "prediction": "...",
  "ground_truth": "...",
  "ground_truth_context": "",
  "seed": 42,
  "run": 1,
  "metrics": {
    "exact_match": 1.0,
    "token_f1": 1.0,
    "answer_correctness_llm": null,
    "lm_judge_justification": null,
    "final_answer_correctness": 1.0,
    "final_answer_correctness_source": "token_f1",
    "context_recall": 0.75,
    "faithfulness": 0.85,
    "hit_rate": null,
    "mrr": null,
    "context_precision": null,
    "ndcg_at_5": null
  },
  "num_contexts": 5,
  "trace": {}
}
```

`context_recall` is computed only when both `ground_truth_answer` and `contexts` are non-empty. `faithfulness` is computed only when both `answer` (the generated prediction) and `contexts` are non-empty. Both are written as `null` otherwise.

**Null aggregation in summary:** `null` values are excluded from the denominator when computing mean/std/median/min/max. If all values for a metric are `null`, the summary entry is `null`. `num_nulls` is tracked per metric per run to make null bias visible (see summary schema below).

#### Summary file (`_summary.json`)

```json
{
  "config": {
    "seed": 42,
    "runs": 5,
    "k_per_type": 5,
    "modes": ["hybrid"],
    "domain_note": "sampling stratified by question_type, not domain — domain distribution is uncontrolled"
  },
  "per_metric": {
    "final_answer_correctness": { "mean": 0.74, "std": 0.03, "median": 0.75, "min": 0.69, "max": 0.79, "num_nulls": 0 },
    "exact_match":              { "mean": 0.68, "std": 0.04, "median": 0.69, "min": 0.62, "max": 0.74, "num_nulls": 0 },
    "token_f1":                 { "mean": 0.74, "std": 0.03, "median": 0.75, "min": 0.69, "max": 0.79, "num_nulls": 0 },
    "answer_correctness_llm":   { "mean": 0.70, "std": 0.05, "median": 0.71, "min": 0.63, "max": 0.76, "num_nulls": 2 },
    "context_recall":           { "mean": 0.51, "std": 0.04, "median": 0.51, "min": 0.46, "max": 0.57, "num_nulls": 0 },
    "faithfulness":             { "mean": 0.81, "std": 0.03, "median": 0.81, "min": 0.76, "max": 0.86, "num_nulls": 0 }
  },
  "per_type": {
    "single_hop":              { "count": 25, "final_answer_correctness": { "mean": 0.81, "std": 0.03 }, "context_recall": { "mean": 0.55, "std": 0.05 } },
    "single_hop_w_conditions": { "count": 25, "final_answer_correctness": { "mean": 0.78, "std": 0.04 }, "context_recall": { "mean": 0.51, "std": 0.05 } },
    "multi_hop":               { "count": 25, "final_answer_correctness": { "mean": 0.74, "std": 0.04 }, "context_recall": { "mean": 0.48, "std": 0.06 } },
    "multi_hop_difficult":     { "count": 25, "final_answer_correctness": { "mean": 0.69, "std": 0.05 }, "context_recall": { "mean": 0.43, "std": 0.07 } },
    "open_ended":              { "count": 25, "final_answer_correctness": { "mean": 0.71, "std": 0.05 }, "context_recall": { "mean": 0.60, "std": 0.06 } },
    "counterfactual":          { "count": 25, "final_answer_correctness": { "mean": 0.69, "std": 0.05 }, "context_recall": { "mean": 0.48, "std": 0.06 } }
  }
}
```

`count` = total non-null samples for that type across all runs. All six question types always appear in `per_type`, even if a type had zero sampled questions (all metrics `null`, `count: 0`).

**Overall** = macro average across question types (equal weight per type, consistent with balanced sampling). See mixed-scale warning in section 2f.

---

### 5. Display

**`print_type_breakdown(aggregated: dict[str, dict[str, Any]])`**
- Input: outer key = `question_type`, inner key = metric name, value = scalar or `None`
- Prints metrics in three labelled sections: `[Extractive: EM / Token F1]`, `[LLM Judge]`, `[Retrieval]`
- One column per question type, one row per metric
- `N/A` for metrics not applicable to that type

**`print_multirun_summary(summary: dict)`**
- Input: the `_summary.json` dict structure defined above
- Prints `mean`, `std`, `median`, `min`, `max`, `num_nulls` for every scalar metric
- Second table: per-type `count` and `std` for `final_answer_correctness`
- Prints `domain_note` from config as a footer warning

**`print_trace_summary()`** — existing method updated to reference `final_answer_correctness` in the correctness column (was `answer_correctness`).

---

## Files Changed

| File | Change |
|---|---|
| `src/ingestion/local_parquet_loader.py` | Replace `max_pairs` with `seed`, `k_per_type`, `stratify` in `load_local_qa_pairs()`; use single `np.random.default_rng(seed)` for all sampling and shuffle |
| `tests/evaluation/hybridrag_eval.py` | Update `load_qa_pairs()` wrapper (remove `max_pairs`, add new params); add `--seed`, `--runs`, `--k-per-type` CLI args; remove `--max-pairs`; add `--output` `.json` extension validation; update `_compute_metrics()` to dispatch on `question_type` and populate new JSON fields; add multi-run loop with per-run file output; add `print_type_breakdown()` and `print_multirun_summary()`; update `print_trace_summary()` to use `final_answer_correctness`; update `print_comparison_table()` replacing its `metric_keys` list with `["exact_match", "token_f1", "answer_correctness_llm", "final_answer_correctness", "context_recall", "faithfulness"]` (removing old `"answer_correctness"`) |
| `tests/evaluation/metrics.py` | Add `normalize()` (NFKD + lowercase + punctuation + articles + whitespace), `calculate_exact_match()`, `calculate_token_f1()` (Counter-based); update LLM judge prompt (structured output, non-overlapping rubric, counterfactual clause, score clamping to [0,1]) |

## Out of Scope

- Changes to the retrieval pipeline (`hybridrag_pipeline.py`, `hybrid_retriever.py`, `graph_retriever.py`)
- Multi-hop retrieval improvements (planned for next phase)
- New ingestion or embedding changes
- Populating `ground_truth_context` — not available in the dataset
