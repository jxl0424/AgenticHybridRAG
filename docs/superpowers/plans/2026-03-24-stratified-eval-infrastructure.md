# Stratified Eval Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential top-of-file QA sampling with stratified random sampling, add extractive metrics (EM + Token F1), update the LLM judge to produce structured output with clamping, update `_compute_metrics()` to dispatch metrics by question type, and add multi-run CLI support with per-type breakdown and summary output.

**Architecture:** Three files change. `metrics.py` gains pure-function extractive metrics and a redesigned LLM judge. `local_parquet_loader.py` gains stratified sampling driven by a single `numpy` RNG. `hybridrag_eval.py` gains type-aware dispatch, multi-run looping, two new display methods, and refactored CLI args.

**Tech Stack:** Python 3.11, pandas, numpy, tenacity, pytest, collections.Counter, unicodedata

---

## File Map

| File | Change type | Responsibility |
|------|-------------|----------------|
| `tests/evaluation/metrics.py` | Modify | Add `normalize()`, `calculate_exact_match()`, `calculate_token_f1()`; update `calculate_answer_correctness()` to return `dict` with structured prompt + clamping |
| `src/ingestion/local_parquet_loader.py` | Modify | Replace `max_pairs` param with `seed`, `k_per_type`, `stratify` in `load_local_qa_pairs()` |
| `tests/evaluation/hybridrag_eval.py` | Modify | Update `load_qa_pairs()` wrapper; update `_compute_metrics()` with type dispatch; update `run()` to accept `seed`/`run`; add `_build_type_breakdown()`, `_build_summary()`, `print_type_breakdown()`, `print_multirun_summary()`; add module-level `_derive_run_path()`, `_derive_summary_path()`, `validate_output_path()`; overhaul CLI |
| `tests/unit/test_metrics_extractive.py` | Create | Tests for `normalize()`, `calculate_exact_match()`, `calculate_token_f1()` |
| `tests/unit/test_metrics_llm_judge.py` | Create | Tests for structured LLM judge, score clamping, parse failure |
| `tests/unit/test_local_qa_loader.py` | Modify | Remove `max_pairs` tests; add stratified sampling tests |
| `tests/unit/test_eval_dispatch.py` | Create | Tests for `_compute_metrics()` type dispatch |
| `tests/unit/test_eval_output.py` | Create | Tests for path helpers, `_build_summary()` aggregation |
| `tests/unit/test_eval_display.py` | Create | Tests for `print_type_breakdown()` and `print_multirun_summary()` |

---

### Task 1: Extractive metrics — normalize, EM, Token F1

**Files:**
- Modify: `tests/evaluation/metrics.py`
- Create: `tests/unit/test_metrics_extractive.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_metrics_extractive.py
import pytest
from tests.evaluation.metrics import RAGMetrics

@pytest.fixture
def m():
    return RAGMetrics()  # no LLM needed

class TestNormalize:
    def test_lowercase(self, m):
        assert m.normalize("Hello World") == "hello world"

    def test_removes_articles(self, m):
        assert m.normalize("the cat sat on an mat") == "cat sat on mat"

    def test_removes_punctuation(self, m):
        assert m.normalize("hello, world!") == "hello world"

    def test_nfkd_accent(self, m):
        assert m.normalize("résumé") == "resume"

    def test_collapses_whitespace(self, m):
        assert m.normalize("  hello   world  ") == "hello world"

    def test_empty_string(self, m):
        assert m.normalize("") == ""


class TestExactMatch:
    def test_identical(self, m):
        assert m.calculate_exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self, m):
        assert m.calculate_exact_match("paris", "Paris") == 1.0

    def test_article_stripped(self, m):
        assert m.calculate_exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_different(self, m):
        assert m.calculate_exact_match("London", "Paris") == 0.0

    def test_empty_both(self, m):
        assert m.calculate_exact_match("", "") == 1.0


class TestTokenF1:
    def test_identical(self, m):
        assert m.calculate_token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_partial_overlap(self, m):
        # normalize removes "on": pred="cat sat", gt="cat mat"
        # common={"cat":1} → precision=1/2, recall=1/2, f1=0.4
        result = m.calculate_token_f1("cat sat", "cat on mat")
        assert result == pytest.approx(0.4, abs=1e-3)

    def test_counter_based_not_set_based(self, m):
        # pred="cat cat dog", gt="cat dog dog"
        # pred_counts={cat:2, dog:1}, gt_counts={cat:1, dog:2}
        # common = min sums = {cat:1, dog:1} = 2
        # precision = 2/3, recall = 2/3, f1 = 2/3
        result = m.calculate_token_f1("cat cat dog", "cat dog dog")
        assert result == pytest.approx(2 / 3, abs=1e-3)

    def test_empty_prediction(self, m):
        assert m.calculate_token_f1("", "Paris") == 0.0

    def test_empty_ground_truth(self, m):
        assert m.calculate_token_f1("Paris", "") == 0.0

    def test_both_empty(self, m):
        assert m.calculate_token_f1("", "") == 1.0
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/unit/test_metrics_extractive.py -v
```
Expected: FAILED — `AttributeError: 'RAGMetrics' object has no attribute 'normalize'`

- [ ] **Step 3: Implement in metrics.py**

Add these three methods to the `RAGMetrics` class (insert before `_extract_key_phrases`):

```python
def normalize(self, text: str) -> str:
    """NFKD → lowercase → strip punctuation → remove articles → collapse whitespace."""
    import unicodedata, re
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)

def calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction == normalized ground truth, else 0.0."""
    return 1.0 if self.normalize(prediction) == self.normalize(ground_truth) else 0.0

def calculate_token_f1(self, prediction: str, ground_truth: str) -> float:
    """Counter-based (bag) token F1, SQuAD-style."""
    from collections import Counter
    pred_tokens = self.normalize(prediction).split()
    gt_tokens = self.normalize(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 4: Run to verify tests pass**

```
pytest tests/unit/test_metrics_extractive.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_metrics_extractive.py tests/evaluation/metrics.py
git commit -m "feat: add normalize, calculate_exact_match, calculate_token_f1 to RAGMetrics"
```

---

### Task 2: Structured LLM judge — clamping, parse failure, counterfactual clause

**Files:**
- Modify: `tests/evaluation/metrics.py`
- Create: `tests/unit/test_metrics_llm_judge.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_metrics_llm_judge.py
import pytest
from unittest.mock import MagicMock, patch
from tests.evaluation.metrics import RAGMetrics


def _make_metrics(judge_response: str) -> RAGMetrics:
    """RAGMetrics with _generate_with_retry mocked to return a fixed string."""
    m = RAGMetrics(llm_client=MagicMock())
    patch.object(m, "_generate_with_retry", return_value=judge_response).start()
    return m


class TestAnswerCorrectnessStructuredOutput:
    def test_returns_dict_with_score_and_justification(self):
        m = _make_metrics("Score: 0.8\nJustification: Answer is mostly correct.")
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert isinstance(result, dict)
        assert "score" in result
        assert "justification" in result

    def test_score_parsed_correctly(self):
        m = _make_metrics("Score: 0.75\nJustification: Partially correct.")
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(0.75)

    def test_score_clamped_above_1(self):
        m = _make_metrics("Score: 1.5\nJustification: Way too high.")
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(1.0)

    def test_score_clamped_below_0(self):
        m = _make_metrics("Score: -0.2\nJustification: Negative.")
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(0.0)

    def test_parse_failure_returns_none_score(self):
        m = _make_metrics("I think it is pretty good overall honestly")
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] is None

    def test_parse_failure_returns_raw_response_in_justification(self):
        raw = "I think it is pretty good overall"
        m = _make_metrics(raw)
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["justification"] is not None
        assert len(result["justification"]) > 0

    def test_counterfactual_prompt_contains_premise_clause(self):
        captured = {}
        m = RAGMetrics(llm_client=MagicMock())

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Score: 0.8\nJustification: Correct."

        with patch.object(m, "_generate_with_retry", side_effect=_capture):
            m.calculate_answer_correctness("q", "gt", "pred", question_type="counterfactual")

        assert "premise is false" in captured["prompt"].lower()

    def test_structured_prompt_requires_score_format(self):
        captured = {}
        m = RAGMetrics(llm_client=MagicMock())

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Score: 0.7\nJustification: OK."

        with patch.object(m, "_generate_with_retry", side_effect=_capture):
            m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")

        assert "Score:" in captured["prompt"]
        assert "Justification:" in captured["prompt"]

    def test_no_llm_returns_none_score(self):
        m = RAGMetrics(llm_client=None)
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] is None
        assert result["justification"] is None
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/unit/test_metrics_llm_judge.py -v
```
Expected: FAILED — `calculate_answer_correctness() takes 4 positional arguments` or returns `float` not `dict`

- [ ] **Step 3: Update calculate_answer_correctness() in metrics.py**

Replace the entire `calculate_answer_correctness` method with:

```python
def calculate_answer_correctness(
    self,
    query: str,
    ground_truth_answer: str,
    generated_answer: str,
    question_type: str = "open_ended",
) -> dict:
    """
    LLM judge for answer quality.

    Returns:
        {"score": float | None, "justification": str | None}
        score is clamped to [0.0, 1.0]; None on parse failure or missing LLM.
    """
    if not self.llm:
        return {"score": None, "justification": None}

    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "i don't know",
        "cannot answer",
        "no information",
    ]
    if any(p in generated_answer.lower() for p in refusal_phrases):
        if len(ground_truth_answer.split()) < 50:
            return {"score": 0.5, "justification": "Model correctly declined to answer."}
        return {"score": 0.0, "justification": "Model refused to answer."}

    counterfactual_clause = ""
    if question_type == "counterfactual":
        counterfactual_clause = (
            "\nIf the model correctly identifies that a premise is false or cannot be "
            "verified from context, treat this as a valid high-scoring response (0.7+)."
        )

    prompt = f"""Task: Evaluate how correct the generated answer is compared to the ground truth.

Query: {query}
Ground Truth: {ground_truth_answer}
Generated Answer: {generated_answer}
{counterfactual_clause}
Evaluate based on:
1. Factual accuracy
2. Completeness
3. Semantic similarity

Output format (REQUIRED):
Score: <float between 0 and 1>
Justification: <one sentence>"""

    try:
        response = self._generate_with_retry(prompt, max_tokens=60)
        score_match = re.search(r"Score:\s*(\d*\.?\d+)", response)
        if not score_match:
            print(f"LLM judge parse failed for: {query[:60]!r}")
            return {"score": None, "justification": response[:200]}
        score = min(1.0, max(0.0, float(score_match.group(1))))
        just_match = re.search(r"Justification:\s*(.+)", response, re.DOTALL)
        justification = just_match.group(1).strip()[:500] if just_match else None
        return {"score": score, "justification": justification}
    except Exception as e:
        print(f"Answer correctness metric error: {e}")
        return {"score": None, "justification": None}
```

- [ ] **Step 4: Run to verify tests pass**

```
pytest tests/unit/test_metrics_llm_judge.py tests/unit/test_metrics_extractive.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_metrics_llm_judge.py tests/evaluation/metrics.py
git commit -m "feat: update LLM judge to return structured dict with score clamping and counterfactual clause"
```

---

### Task 3: Stratified sampling in LocalParquetLoader

**Files:**
- Modify: `src/ingestion/local_parquet_loader.py`
- Modify: `tests/unit/test_local_qa_loader.py`

- [ ] **Step 1: Add stratified fixture and new tests; mark max_pairs tests for removal**

In `tests/unit/test_local_qa_loader.py`, add after the existing `qa_data_dir` fixture:

```python
QUESTION_TYPES = [
    "single_hop", "single_hop_w_conditions", "multi_hop",
    "multi_hop_difficult", "open_ended", "counterfactual",
]

@pytest.fixture
def stratified_qa_data_dir(tmp_path):
    """18 test rows: 3 per question type, single domain."""
    rows = []
    for i, qt in enumerate(QUESTION_TYPES):
        for j in range(3):
            rows.append({
                "domain": "arxiv_ai", "split": "test",
                "question_id": i * 10 + j,
                "question": f"Q {qt} {j}?",
                "answer": f"A {qt} {j}.",
                "question_type": qt,
            })
    _make_qa_parquet(tmp_path, "arxiv_ai", rows)
    return str(tmp_path)
```

Then add this test class:

```python
class TestStratifiedSampling:

    def test_k_per_type_returns_correct_count(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        pairs = loader.load_local_qa_pairs(k_per_type=2, seed=42)
        # 6 types × 2 = 12 pairs
        assert len(pairs) == 12

    def test_all_question_types_represented(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        pairs = loader.load_local_qa_pairs(k_per_type=2, seed=42)
        types_found = {p["question_type"] for p in pairs}
        assert types_found == set(QUESTION_TYPES)

    def test_same_seed_reproduces_same_pairs(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        run1 = loader.load_local_qa_pairs(k_per_type=2, seed=99)
        run2 = loader.load_local_qa_pairs(k_per_type=2, seed=99)
        assert [p["question"] for p in run1] == [p["question"] for p in run2]

    def test_different_seeds_produce_different_ordering(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        run1 = loader.load_local_qa_pairs(k_per_type=2, seed=1)
        run2 = loader.load_local_qa_pairs(k_per_type=2, seed=2)
        # Final shuffle must differ between seeds
        assert [p["question"] for p in run1] != [p["question"] for p in run2]

    def test_warns_when_type_smaller_than_k(self, stratified_qa_data_dir, capsys):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        # k_per_type=5 but only 3 rows per type → warning
        loader.load_local_qa_pairs(k_per_type=5, seed=0)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_stratify_false_samples_from_full_pool(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        pairs = loader.load_local_qa_pairs(k_per_type=2, seed=42, stratify=False)
        # stratify=False: sample k_per_type*6=12 from full pool
        assert len(pairs) == 12

    def test_returned_dicts_have_required_keys(self, stratified_qa_data_dir):
        loader = LocalParquetLoader(stratified_qa_data_dir)
        pairs = loader.load_local_qa_pairs(k_per_type=1, seed=0)
        for p in pairs:
            assert set(p.keys()) >= {"question", "ground_truth_answer", "ground_truth_context",
                                      "question_type", "domain"}
```

Also, delete `test_max_pairs_limits_results` from `TestLoadLocalQAPairs` (lines 101-104 of the original file).

- [ ] **Step 2: Run new tests to verify they fail**

```
pytest tests/unit/test_local_qa_loader.py::TestStratifiedSampling -v
```
Expected: FAILED — `load_local_qa_pairs() got unexpected keyword argument 'k_per_type'`

- [ ] **Step 3: Replace load_local_qa_pairs() in local_parquet_loader.py**

Replace the entire `load_local_qa_pairs` method with:

```python
def load_local_qa_pairs(
    self,
    domains: list[str] | None = None,
    split: str = "test",
    k_per_type: int = 5,
    seed: int | None = None,
    stratify: bool = True,
) -> list[dict]:
    """
    Load QA pairs from text_qa/{domain}/qa.parquet with stratified random sampling.

    Args:
        domains: Domains to load (default: all three arxiv_* domains).
        split: Dataset split to filter on ("test" or "train").
        k_per_type: Samples per question_type per run.
        seed: numpy RNG seed. If None, a random seed is chosen (caller should print it).
        stratify: If True, sample k_per_type per question_type.
                  If False, sample k_per_type*6 from the full pool.

    Returns:
        Shuffled list of dicts with keys:
            question, ground_truth_answer, ground_truth_context, question_type, domain
    """
    import pandas as pd
    import numpy as np

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))

    rng = np.random.default_rng(seed)

    if domains is None:
        domains = DOMAINS

    all_dfs: list[pd.DataFrame] = []
    for domain in domains:
        path = self.data_dir / f"text_qa/{domain}/qa.parquet"
        if not path.exists():
            logger.warning(f"QA parquet not found, skipping: {path}")
            continue
        df = pd.read_parquet(path)
        if "split" in df.columns:
            df = df[df["split"] == split]
        df = df[df["question"].notna() & df["answer"].notna()]
        df = df[df["question"].astype(str).str.strip() != ""]
        df = df[df["answer"].astype(str).str.strip() != ""]
        df["domain"] = domain
        all_dfs.append(df)

    if not all_dfs:
        return []

    full_df = pd.concat(all_dfs, ignore_index=True)

    QUESTION_TYPES = [
        "single_hop", "single_hop_w_conditions", "multi_hop",
        "multi_hop_difficult", "open_ended", "counterfactual",
    ]

    if not stratify:
        n = min(k_per_type * len(QUESTION_TYPES), len(full_df))
        sampled = full_df.sample(n=n, random_state=rng)
    else:
        sampled_groups: list[pd.DataFrame] = []
        for qt in QUESTION_TYPES:
            group_df = full_df[full_df["question_type"] == qt]
            if len(group_df) == 0:
                continue
            if len(group_df) < k_per_type:
                print(f"WARNING: only {len(group_df)} rows available for '{qt}', using all")
                sampled_groups.append(group_df)
            else:
                sampled_groups.append(
                    group_df.sample(n=k_per_type, random_state=rng)
                )
        if not sampled_groups:
            return []
        sampled = pd.concat(sampled_groups, ignore_index=True)

    # Final shuffle — same RNG, no reset
    idx = rng.permutation(len(sampled))
    sampled = sampled.iloc[idx].reset_index(drop=True)

    pairs = []
    for _, row in sampled.iterrows():
        pairs.append({
            "question": str(row["question"]).strip(),
            "ground_truth_answer": str(row["answer"]).strip(),
            "ground_truth_context": "",
            "question_type": str(row.get("question_type") or ""),
            "domain": str(row.get("domain") or ""),
        })

    logger.info(
        f"Loaded {len(pairs)} QA pairs "
        f"(split={split}, seed={seed}, k_per_type={k_per_type}, stratify={stratify})"
    )
    return pairs
```

- [ ] **Step 4: Run all loader tests to verify they pass**

```
pytest tests/unit/test_local_qa_loader.py -v
```
Expected: All PASSED (the `test_max_pairs_limits_results` test was deleted; `test_load_qa_pairs_local_max_pairs_forwarded` in `TestHybridRAGEvaluatorLocalLoad` will fail — fix in next step)

- [ ] **Step 5: Update the HybridRAGEvaluator wrapper test**

In `TestHybridRAGEvaluatorLocalLoad`, rename `test_load_qa_pairs_local_max_pairs_forwarded` to test the new `seed` parameter instead:

```python
def test_load_qa_pairs_local_seed_forwarded(self, stratified_qa_data_dir):
    """seed is forwarded to the local loader."""
    from unittest.mock import MagicMock

    HybridRAGEvaluator = _import_evaluator()
    mock_pipeline = MagicMock()
    mock_pipeline.llm = MagicMock()
    evaluator = HybridRAGEvaluator(mock_pipeline)

    run1 = evaluator.load_qa_pairs(local_data_dir=stratified_qa_data_dir, seed=7, k_per_type=2)
    run2 = evaluator.load_qa_pairs(local_data_dir=stratified_qa_data_dir, seed=7, k_per_type=2)
    assert [p["question"] for p in run1] == [p["question"] for p in run2]
```

Note: `stratified_qa_data_dir` is defined at the top of `test_local_qa_loader.py`. Since this test class is in the same file, it is already accessible via pytest's fixture discovery — no extra import needed.

- [ ] **Step 6: Run all loader tests**

```
pytest tests/unit/test_local_qa_loader.py -v
```
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/local_parquet_loader.py tests/unit/test_local_qa_loader.py
git commit -m "feat: replace max_pairs with stratified sampling (seed, k_per_type, stratify) in load_local_qa_pairs"
```

---

### Task 4: Type-aware metric dispatch in _compute_metrics()

**Files:**
- Modify: `tests/evaluation/hybridrag_eval.py`
- Create: `tests/unit/test_eval_dispatch.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_eval_dispatch.py
"""
Tests for HybridRAGEvaluator._compute_metrics() type-based dispatch.
"""
import sys
from unittest.mock import MagicMock
import pytest


def _make_evaluator():
    """Import HybridRAGEvaluator with all heavy deps stubbed."""
    for mod in (
        "src.ingestion.hf_hybridrag_loader",
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.graph.knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
        "src.observability.tracer",
    ):
        sys.modules.setdefault(mod, MagicMock())

    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock()
    evaluator = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
    evaluator.pipeline = mock_pipeline
    evaluator.tracer = None

    # Inject a controllable metrics mock
    metrics_mock = MagicMock()
    metrics_mock.normalize.return_value = "normalized prediction text"
    metrics_mock.calculate_exact_match.return_value = 0.0
    metrics_mock.calculate_token_f1.return_value = 0.0
    metrics_mock.calculate_answer_correctness.return_value = {
        "score": 0.8, "justification": "Correct."
    }
    metrics_mock.calculate_context_recall.return_value = 0.6
    metrics_mock.calculate_faithfulness.return_value = 0.7
    evaluator.metrics = metrics_mock
    return evaluator


RESPONSE = {"answer": "some prediction", "contexts": ["ctx1", "ctx2"], "trace": {}}


class TestDispatchSingleHop:
    def test_computes_em_and_token_f1(self):
        ev = _make_evaluator()
        ev.metrics.calculate_exact_match.return_value = 1.0
        ev.metrics.calculate_token_f1.return_value = 1.0
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["metrics"]["exact_match"] == 1.0
        assert item["metrics"]["token_f1"] == 1.0

    def test_llm_judge_not_called(self):
        ev = _make_evaluator()
        ev._compute_metrics("q", "gt", "", RESPONSE,
                             question_type="single_hop", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()

    def test_final_answer_correctness_source_is_token_f1(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.75
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["metrics"]["final_answer_correctness"] == pytest.approx(0.75)
        assert item["metrics"]["final_answer_correctness_source"] == "token_f1"


class TestDispatchOpenEnded:
    def test_calls_llm_judge(self):
        ev = _make_evaluator()
        ev._compute_metrics("q", "gt", "", RESPONSE,
                             question_type="open_ended", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_called_once()

    def test_em_and_token_f1_are_none(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="open_ended", domain="d", seed=1, run=1)
        assert item["metrics"]["exact_match"] is None
        assert item["metrics"]["token_f1"] is None

    def test_final_answer_correctness_source_is_llm(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="open_ended", domain="d", seed=1, run=1)
        assert item["metrics"]["final_answer_correctness_source"] == "llm"


class TestDispatchMultiHopDifficult:
    def test_escalates_to_llm_when_token_f1_zero_and_long_prediction(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.0
        # normalize returns 16 words → triggers escalation
        ev.metrics.normalize.return_value = " ".join(["word"] * 16)
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_called_once()
        assert item["metrics"]["final_answer_correctness_source"] == "llm"

    def test_no_escalation_when_token_f1_nonzero(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.5
        ev.metrics.normalize.return_value = " ".join(["word"] * 16)
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()
        assert item["metrics"]["final_answer_correctness_source"] == "token_f1"

    def test_no_escalation_when_prediction_short(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.0
        # Only 5 words → no escalation (threshold is > 15)
        ev.metrics.normalize.return_value = "five short words only here"
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()


class TestOutputSchema:
    def test_output_has_required_keys(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="arxiv_ai", seed=42, run=2)
        for key in ("question", "question_type", "domain", "prediction", "ground_truth",
                    "ground_truth_context", "seed", "run", "metrics", "num_contexts", "trace"):
            assert key in item, f"Missing key: {key}"

    def test_domain_and_seed_in_output(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="arxiv_cy", seed=77, run=3)
        assert item["domain"] == "arxiv_cy"
        assert item["seed"] == 77
        assert item["run"] == 3

    def test_prediction_renamed_from_answer(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["prediction"] == "some prediction"
        assert "answer" not in item  # old key must be gone
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/unit/test_eval_dispatch.py -v
```
Expected: FAILED — `_compute_metrics() takes 5 positional arguments` (missing new params)

- [ ] **Step 3: Replace _compute_metrics() in hybridrag_eval.py**

Replace the entire `_compute_metrics` method with:

```python
def _compute_metrics(
    self,
    question: str,
    gt_answer: str,
    gt_context: str,
    response: dict,
    question_type: str = "",
    domain: str = "",
    seed: int = 0,
    run: int = 1,
) -> dict:
    """Compute metrics for one QA pair, dispatching by question_type."""
    contexts = response.get("contexts", [])
    answer = response.get("answer", "")

    metrics = {
        "exact_match": None,
        "token_f1": None,
        "answer_correctness_llm": None,
        "lm_judge_justification": None,
        "final_answer_correctness": None,
        "final_answer_correctness_source": None,
        "context_recall": None,
        "faithfulness": None,
        "hit_rate": None,
        "mrr": None,
        "context_precision": None,
        "ndcg_at_5": None,
    }

    if gt_answer and answer:
        if question_type in self.EXTRACTIVE_TYPES:
            em = self.metrics.calculate_exact_match(answer, gt_answer)
            tf1 = self.metrics.calculate_token_f1(answer, gt_answer)
            metrics["exact_match"] = em
            metrics["token_f1"] = tf1
            metrics["final_answer_correctness"] = tf1
            metrics["final_answer_correctness_source"] = "token_f1"

        elif question_type == "multi_hop_difficult":
            em = self.metrics.calculate_exact_match(answer, gt_answer)
            tf1 = self.metrics.calculate_token_f1(answer, gt_answer)
            metrics["exact_match"] = em
            metrics["token_f1"] = tf1
            pred_len = len(self.metrics.normalize(answer).split())
            if tf1 == 0.0 and pred_len > 15:
                judge = self.metrics.calculate_answer_correctness(
                    question, gt_answer, answer, question_type="multi_hop_difficult"
                )
                metrics["answer_correctness_llm"] = judge["score"]
                metrics["lm_judge_justification"] = judge["justification"]
                metrics["final_answer_correctness"] = judge["score"]
                metrics["final_answer_correctness_source"] = "llm" if judge["score"] is not None else None
            else:
                metrics["final_answer_correctness"] = tf1
                metrics["final_answer_correctness_source"] = "token_f1"

        elif question_type in self.GENERATIVE_TYPES:
            judge = self.metrics.calculate_answer_correctness(
                question, gt_answer, answer, question_type=question_type
            )
            metrics["answer_correctness_llm"] = judge["score"]
            metrics["lm_judge_justification"] = judge["justification"]
            metrics["final_answer_correctness"] = judge["score"]
            metrics["final_answer_correctness_source"] = "llm" if judge["score"] is not None else None

    if gt_answer and contexts:
        metrics["context_recall"] = self.metrics.calculate_context_recall(gt_answer, contexts)

    if answer and contexts:
        metrics["faithfulness"] = self.metrics.calculate_faithfulness(question, contexts, answer)

    return {
        "question": question,
        "question_type": question_type,
        "domain": domain,
        "prediction": answer,
        "ground_truth": gt_answer,
        "ground_truth_context": gt_context,
        "seed": seed,
        "run": run,
        "metrics": metrics,
        "num_contexts": len(contexts),
        "trace": response.get("trace", {}),
    }
```

Also add as class-level constants on `HybridRAGEvaluator` (before `__init__`):

```python
EXTRACTIVE_TYPES = frozenset({"single_hop", "single_hop_w_conditions", "multi_hop"})
GENERATIVE_TYPES = frozenset({"open_ended", "counterfactual"})
```

Also update the `run()` loop's call from:
```python
item = self._compute_metrics(question, gt_answer, gt_context, response)
item["mode"] = mode
```
to:
```python
item = self._compute_metrics(
    question, gt_answer, gt_context, response,
    question_type=qa.get("question_type", ""),
    domain=qa.get("domain", ""),
    seed=kwargs.get("seed", 0),
    run=kwargs.get("run", 1),
)
item["mode"] = mode
```

And add `**kwargs` to `run()` for now (replaced in Task 5).

Also update `print_trace_summary` to use `item.get("prediction", item.get("answer", ""))` and `item.get("metrics", {}).get("final_answer_correctness")` for the `corr` column.

Also update `print_comparison_table` metric_keys to:
```python
metric_keys = [
    "exact_match", "token_f1", "answer_correctness_llm",
    "final_answer_correctness", "context_recall", "faithfulness",
]
```

Also update `load_qa_pairs()` wrapper to accept and forward new params:

```python
def load_qa_pairs(
    self,
    split: str = "test",
    local_data_dir: str = "data/hybridrag",
    seed: int | None = None,
    k_per_type: int = 5,
    stratify: bool = True,
) -> list[dict]:
    from src.ingestion.local_parquet_loader import LocalParquetLoader
    loader = LocalParquetLoader(local_data_dir)
    return loader.load_local_qa_pairs(
        split=split, seed=seed, k_per_type=k_per_type, stratify=stratify
    )
```

- [ ] **Step 4: Run dispatch tests**

```
pytest tests/unit/test_eval_dispatch.py -v
```
Expected: All PASSED

- [ ] **Step 5: Run full unit suite to check for regressions**

```
pytest tests/unit/ -v --ignore=tests/unit/test_integration_real.py -x
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_eval_dispatch.py tests/evaluation/hybridrag_eval.py
git commit -m "feat: add type-aware metric dispatch to _compute_metrics() with new output schema"
```

---

### Task 5: Multi-run CLI — path helpers, loop, summary, display methods

**Files:**
- Modify: `tests/evaluation/hybridrag_eval.py`
- Create: `tests/unit/test_eval_output.py`
- Create: `tests/unit/test_eval_display.py`

- [ ] **Step 1: Write failing tests for path helpers and summary**

```python
# tests/unit/test_eval_output.py
"""Tests for multi-run path helpers and summary aggregation."""
import sys
from unittest.mock import MagicMock
import pytest


def _setup_stubs():
    for mod in (
        "src.ingestion.hf_hybridrag_loader",
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.graph.knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
        "src.observability.tracer",
    ):
        sys.modules.setdefault(mod, MagicMock())


_setup_stubs()

from tests.evaluation.hybridrag_eval import (
    _derive_run_path,
    _derive_summary_path,
    validate_output_path,
    HybridRAGEvaluator,
)


class TestPathHelpers:
    def test_derive_run_path_run1(self):
        assert _derive_run_path("tests/results/eval.json", 1) == "tests/results/eval_run1.json"

    def test_derive_run_path_run3(self):
        assert _derive_run_path("tests/results/eval.json", 3) == "tests/results/eval_run3.json"

    def test_derive_summary_path(self):
        assert _derive_summary_path("tests/results/eval.json") == "tests/results/eval_summary.json"

    def test_validate_output_path_accepts_json(self):
        validate_output_path("my_eval.json")  # must not raise

    def test_validate_output_path_rejects_non_json(self):
        with pytest.raises(ValueError, match=".json"):
            validate_output_path("my_eval.txt")


class TestBuildSummary:
    def _make_evaluator(self):
        ev = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
        ev.pipeline = MagicMock()
        ev.tracer = None
        ev.metrics = MagicMock()
        return ev

    def _make_items(self):
        """Six items: one per question type, with known metric values."""
        types = ["single_hop", "single_hop_w_conditions", "multi_hop",
                 "multi_hop_difficult", "open_ended", "counterfactual"]
        items = []
        for qt in types:
            items.append({
                "question_type": qt,
                "metrics": {
                    "final_answer_correctness": 0.8,
                    "exact_match": 1.0 if qt not in {"open_ended", "counterfactual"} else None,
                    "token_f1": 0.9 if qt not in {"open_ended", "counterfactual"} else None,
                    "answer_correctness_llm": 0.8 if qt in {"open_ended", "counterfactual"} else None,
                    "context_recall": 0.6,
                    "faithfulness": 0.7,
                },
            })
        return items

    def test_summary_has_required_top_level_keys(self):
        ev = self._make_evaluator()
        summary = ev._build_summary(self._make_items(), {"seed": 42, "runs": 1,
                                                          "k_per_type": 5, "modes": ["hybrid"]})
        assert "config" in summary
        assert "per_metric" in summary
        assert "per_type" in summary

    def test_config_contains_domain_note(self):
        ev = self._make_evaluator()
        summary = ev._build_summary([], {"seed": 0, "runs": 1, "k_per_type": 5, "modes": []})
        assert "domain" in summary["config"]["domain_note"].lower()

    def test_per_metric_contains_final_answer_correctness(self):
        ev = self._make_evaluator()
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        assert "final_answer_correctness" in summary["per_metric"]

    def test_per_metric_num_nulls_counts_nulls(self):
        ev = self._make_evaluator()
        items = self._make_items()
        # 2 items have answer_correctness_llm=None (extractive types)
        summary = ev._build_summary(items, {"seed": 1, "runs": 1, "k_per_type": 5, "modes": []})
        null_count = summary["per_metric"]["answer_correctness_llm"]["num_nulls"]
        assert null_count == 4  # single_hop, single_hop_w_conditions, multi_hop, multi_hop_difficult

    def test_per_type_contains_all_six_types(self):
        ev = self._make_evaluator()
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        expected = {"single_hop", "single_hop_w_conditions", "multi_hop",
                    "multi_hop_difficult", "open_ended", "counterfactual"}
        assert set(summary["per_type"].keys()) == expected

    def test_per_type_count_reflects_non_null_final_ac(self):
        ev = self._make_evaluator()
        # All items have final_answer_correctness=0.8 → count=1 per type
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        for qt in summary["per_type"]:
            assert summary["per_type"][qt]["count"] == 1
```

```python
# tests/unit/test_eval_display.py
"""Tests for print_type_breakdown() and print_multirun_summary()."""
import sys
from unittest.mock import MagicMock
import pytest


def _setup_stubs():
    for mod in (
        "src.ingestion.hf_hybridrag_loader",
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.graph.knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
        "src.observability.tracer",
    ):
        sys.modules.setdefault(mod, MagicMock())


_setup_stubs()

from tests.evaluation.hybridrag_eval import HybridRAGEvaluator


@pytest.fixture
def ev():
    e = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
    e.pipeline = MagicMock()
    e.tracer = None
    e.metrics = MagicMock()
    return e


class TestPrintTypeBreakdown:
    def test_contains_extractive_section_header(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "[Extractive: EM / Token F1]" in capsys.readouterr().out

    def test_contains_llm_judge_section_header(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "[LLM Judge]" in capsys.readouterr().out

    def test_shows_na_for_missing_type(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "N/A" in capsys.readouterr().out

    def test_shows_score_for_known_type(self, ev, capsys):
        ev.print_type_breakdown({"single_hop": {"final_answer_correctness": 0.85,
                                                  "context_recall": 0.60}})
        assert "0.8500" in capsys.readouterr().out


class TestPrintMultirunSummary:
    SUMMARY = {
        "config": {"seed": 42, "runs": 3, "k_per_type": 5,
                   "modes": ["hybrid"], "domain_note": "domain imbalance note"},
        "per_metric": {
            "final_answer_correctness": {
                "mean": 0.74, "std": 0.03, "median": 0.75,
                "min": 0.69, "max": 0.79, "num_nulls": 0,
            },
        },
        "per_type": {
            "single_hop": {
                "count": 15,
                "final_answer_correctness": {"mean": 0.81, "std": 0.03},
                "context_recall": {"mean": 0.55, "std": 0.05},
            },
        },
    }

    def test_contains_multi_run_summary_header(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        assert "Multi-Run Summary" in capsys.readouterr().out

    def test_shows_seed_and_runs(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        assert "42" in out
        assert "3" in out

    def test_shows_null_count_column(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        # Column header should contain "nulls" or "num_nulls"
        assert "null" in out.lower()

    def test_shows_per_type_mean(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        assert "0.8100" in capsys.readouterr().out

    def test_shows_domain_note_warning(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        assert "WARNING" in out or "domain" in out.lower()
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/unit/test_eval_output.py tests/unit/test_eval_display.py -v
```
Expected: FAILED — `cannot import name '_derive_run_path'`

- [ ] **Step 3: Add module-level helpers and new methods to hybridrag_eval.py**

Add immediately after the imports, before the `HybridRAGEvaluator` class:

```python
# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def validate_output_path(path: str) -> None:
    """Raise ValueError if path does not end in .json."""
    if not path.endswith(".json"):
        raise ValueError(f"--output must end in .json, got: {path!r}")


def _derive_run_path(output: str, run_index: int) -> str:
    """'my_eval.json' → 'my_eval_run1.json'"""
    p = Path(output)
    return str(p.parent / f"{p.stem}_run{run_index}{p.suffix}")


def _derive_summary_path(output: str) -> str:
    """'my_eval.json' → 'my_eval_summary.json'"""
    p = Path(output)
    return str(p.parent / f"{p.stem}_summary{p.suffix}")
```

Add to `HybridRAGEvaluator`:

```python
def _build_type_breakdown(self, raw_results: dict) -> dict:
    """
    Compute per-type averages of final_answer_correctness and context_recall.

    Args:
        raw_results: {mode: [list of per-question items]} as saved to JSON.

    Returns:
        {question_type: {"final_answer_correctness": float|None, "context_recall": float|None}}
    """
    by_type: dict[str, list[dict]] = {}
    for mode_items in raw_results.values():
        for item in mode_items:
            qt = item.get("question_type", "unknown")
            by_type.setdefault(qt, []).append(item.get("metrics", {}))

    result = {}
    for qt, metric_list in by_type.items():
        type_entry = {}
        for key in ("final_answer_correctness", "context_recall"):
            values = [m.get(key) for m in metric_list if m.get(key) is not None]
            type_entry[key] = round(sum(values) / len(values), 4) if values else None
        result[qt] = type_entry
    return result

def _build_summary(self, all_items: list[dict], config: dict) -> dict:
    """
    Aggregate metrics across all runs for the multi-run summary JSON.

    Args:
        all_items: Flat list of per-question items from all runs.
        config: {seed, runs, k_per_type, modes}.

    Returns:
        {"config": {...}, "per_metric": {...}, "per_type": {...}}
    """
    import statistics

    METRIC_KEYS = [
        "final_answer_correctness", "exact_match", "token_f1",
        "answer_correctness_llm", "context_recall", "faithfulness",
    ]
    QUESTION_TYPES = [
        "single_hop", "single_hop_w_conditions", "multi_hop",
        "multi_hop_difficult", "open_ended", "counterfactual",
    ]

    per_metric = {}
    for key in METRIC_KEYS:
        values = [
            item["metrics"][key]
            for item in all_items
            if isinstance(item.get("metrics", {}).get(key), float)
        ]
        num_nulls = sum(
            1 for item in all_items
            if not isinstance(item.get("metrics", {}).get(key), float)
        )
        if values:
            per_metric[key] = {
                "mean": round(sum(values) / len(values), 4),
                "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "num_nulls": num_nulls,
            }
        else:
            per_metric[key] = {"mean": None, "std": None, "median": None,
                                "min": None, "max": None, "num_nulls": num_nulls}

    per_type = {}
    for qt in QUESTION_TYPES:
        type_items = [item for item in all_items if item.get("question_type") == qt]
        fac_vals = [
            item["metrics"]["final_answer_correctness"]
            for item in type_items
            if isinstance(item.get("metrics", {}).get("final_answer_correctness"), float)
        ]
        cr_vals = [
            item["metrics"]["context_recall"]
            for item in type_items
            if isinstance(item.get("metrics", {}).get("context_recall"), float)
        ]
        per_type[qt] = {
            "count": len(fac_vals),
            "final_answer_correctness": {
                "mean": round(sum(fac_vals) / len(fac_vals), 4) if fac_vals else None,
                "std": round(statistics.stdev(fac_vals) if len(fac_vals) > 1 else 0.0, 4) if fac_vals else None,
            },
            "context_recall": {
                "mean": round(sum(cr_vals) / len(cr_vals), 4) if cr_vals else None,
                "std": round(statistics.stdev(cr_vals) if len(cr_vals) > 1 else 0.0, 4) if cr_vals else None,
            },
        }

    return {
        "config": {
            **config,
            "domain_note": (
                "Sampling is stratified by question_type, not domain — "
                "domain distribution is uncontrolled."
            ),
        },
        "per_metric": per_metric,
        "per_type": per_type,
    }

def print_type_breakdown(self, aggregated: dict) -> None:
    """
    Print final_answer_correctness and context_recall per question type.

    Args:
        aggregated: {question_type: {"final_answer_correctness": float|None, ...}}
    """
    print("\n" + "=" * 80)
    print("Breakdown by Question Type")
    print("=" * 80)

    def _fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    print("\n[Extractive: EM / Token F1]")
    for qt in ("single_hop", "single_hop_w_conditions", "multi_hop", "multi_hop_difficult"):
        data = aggregated.get(qt, {})
        print(f"  {qt:<32} final_ac={_fmt(data.get('final_answer_correctness'))}  "
              f"context_recall={_fmt(data.get('context_recall'))}")

    print("\n[LLM Judge]")
    for qt in ("open_ended", "counterfactual"):
        data = aggregated.get(qt, {})
        print(f"  {qt:<32} final_ac={_fmt(data.get('final_answer_correctness'))}  "
              f"context_recall={_fmt(data.get('context_recall'))}")

    print("=" * 80)

def print_multirun_summary(self, summary: dict) -> None:
    """Print multi-run aggregated summary to stdout."""
    cfg = summary["config"]
    print("\n" + "=" * 88)
    print(f"Multi-Run Summary  ({cfg['runs']} runs, seed={cfg['seed']}, "
          f"k_per_type={cfg['k_per_type']})")
    print("=" * 88)

    # Per-metric table
    hdr = f"{'Metric':<32} {'mean':>8} {'std':>8} {'median':>8} {'min':>8} {'max':>8} {'nulls':>6}"
    print(hdr)
    print("-" * len(hdr))
    for key, stats in summary["per_metric"].items():
        if stats is None or stats.get("mean") is None:
            print(f"{key:<32} {'N/A':>8}")
            continue
        print(
            f"{key:<32} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
            f"{stats['median']:>8.4f} {stats['min']:>8.4f} {stats['max']:>8.4f} "
            f"{stats['num_nulls']:>6}"
        )

    # Per-type table
    print(f"\n{'Type':<32} {'count':>6} {'mean_final_ac':>14} {'std':>8}")
    print("-" * 62)
    for qt, data in summary["per_type"].items():
        fac = data["final_answer_correctness"]
        if fac and fac["mean"] is not None:
            print(f"{qt:<32} {data['count']:>6} {fac['mean']:>14.4f} {fac['std']:>8.4f}")
        else:
            print(f"{qt:<32} {data['count']:>6} {'N/A':>14}")

    print(f"\nWARNING: {cfg['domain_note']}")
    print("=" * 88)
```

- [ ] **Step 4: Run to verify tests pass**

```
pytest tests/unit/test_eval_output.py tests/unit/test_eval_display.py -v
```
Expected: All PASSED

- [ ] **Step 5: Replace CLI entry point in hybridrag_eval.py**

Replace everything from `if __name__ == "__main__":` to end of file with:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HybridRAG-Bench pipeline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base RNG seed. If omitted, a random seed is chosen and printed.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent eval runs (seeds: [seed, seed+1, ...])")
    parser.add_argument("--k-per-type", type=int, default=5,
                        help="Questions sampled per question_type per run")
    parser.add_argument("--top-k", type=int, default=5, help="Contexts per query")
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid"])
    parser.add_argument("--output", default="tests/results/hybridrag_eval.json",
                        help="Output path (must end in .json)")
    parser.add_argument("--local-data-dir", default="data/hybridrag")
    args = parser.parse_args()

    validate_output_path(args.output)

    import numpy as np
    base_seed = args.seed if args.seed is not None else int(
        np.random.default_rng().integers(0, 1_000_000)
    )
    print(f"Seed: {base_seed}")

    print("Initializing pipeline...")
    pipeline = HybridRAGBenchPipeline()
    evaluator = HybridRAGEvaluator(pipeline)

    all_run_items: list[dict] = []

    for run_i in range(1, args.runs + 1):
        seed_i = base_seed + (run_i - 1)
        output_i = _derive_run_path(args.output, run_i) if args.runs > 1 else args.output

        qa_pairs = evaluator.load_qa_pairs(
            local_data_dir=args.local_data_dir,
            seed=seed_i,
            k_per_type=args.k_per_type,
        )

        if not qa_pairs:
            print("No QA pairs found. Check --local-data-dir path and parquet schema.")
            sys.exit(1)

        if args.runs > 1:
            print(f"\n--- Run {run_i}/{args.runs} (seed={seed_i}, {len(qa_pairs)} pairs) ---")

        aggregated = evaluator.run(
            qa_pairs,
            modes=args.modes,
            top_k=args.top_k,
            output_path=output_i,
            seed=seed_i,
            run=run_i,
        )

        with open(output_i) as f:
            raw_results = json.load(f)
        for mode_items in raw_results.values():
            all_run_items.extend(mode_items)

        if args.runs == 1:
            evaluator.print_comparison_table(aggregated)
            type_breakdown = evaluator._build_type_breakdown(raw_results)
            evaluator.print_type_breakdown(type_breakdown)
            evaluator.print_trace_summary(raw_results)

    if args.runs > 1:
        summary = evaluator._build_summary(
            all_run_items,
            {
                "seed": base_seed,
                "runs": args.runs,
                "k_per_type": args.k_per_type,
                "modes": args.modes,
            },
        )
        summary_path = _derive_summary_path(args.output)
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        evaluator.print_multirun_summary(summary)

    if evaluator.tracer is not None:
        print("\nPhoenix is running at http://localhost:6006")
        print("Press Enter to shut down Phoenix and exit...")
        input()
```

Also update `run()` to accept `seed` and `run` explicitly (replacing the `**kwargs` hack from Task 4):

```python
def run(
    self,
    qa_pairs: list[dict],
    modes: list[str] = None,
    top_k: int = 5,
    output_path: str = "tests/results/hybridrag_eval.json",
    seed: int = 0,
    run: int = 1,
) -> dict:
```

And update the `_compute_metrics` call inside `run()`:

```python
item = self._compute_metrics(
    question, gt_answer, gt_context, response,
    question_type=qa.get("question_type", ""),
    domain=qa.get("domain", ""),
    seed=seed,
    run=run,
)
```

Also update `print_trace_summary` to use the new schema field names (`prediction` not `answer`, `final_answer_correctness` not `answer_correctness`):

```python
# In print_trace_summary, replace:
corr = m.get("answer_correctness", "")
# with:
corr = m.get("final_answer_correctness", "")
# And:
q = item.get("question", "")[:45]
# stays the same; but:
# answer is now "prediction" in item dict — this only affects trace display if needed
```

- [ ] **Step 6: Run full test suite**

```
pytest tests/unit/ -v -x
```
Expected: All PASSED

- [ ] **Step 7: Smoke-test the CLI help**

```
python tests/evaluation/hybridrag_eval.py --help
```
Expected: Shows `--seed`, `--runs`, `--k-per-type`, `--top-k`, `--modes`, `--output`, `--local-data-dir`

- [ ] **Step 8: Commit**

```bash
git add tests/evaluation/hybridrag_eval.py tests/unit/test_eval_output.py tests/unit/test_eval_display.py
git commit -m "feat: add multi-run CLI (--seed, --runs, --k-per-type), path helpers, _build_summary, print_type_breakdown, print_multirun_summary"
```

---

## Usage After Implementation

**Single run (30 questions, reproducible):**
```bash
python tests/evaluation/hybridrag_eval.py \
    --seed 42 --k-per-type 5 --modes hybrid \
    --output tests/results/eval_hybrid.json
```
Outputs: `tests/results/eval_hybrid.json` + console table + type breakdown

**Three-run average:**
```bash
python tests/evaluation/hybridrag_eval.py \
    --seed 42 --runs 3 --k-per-type 5 --modes hybrid \
    --output tests/results/eval_hybrid.json
```
Outputs: `eval_hybrid_run1.json`, `eval_hybrid_run2.json`, `eval_hybrid_run3.json`, `eval_hybrid_summary.json`

**Summary JSON structure:**
```json
{
  "config": {"seed": 42, "runs": 3, "k_per_type": 5, "modes": ["hybrid"],
             "domain_note": "Sampling is stratified by question_type..."},
  "per_metric": {
    "final_answer_correctness": {"mean": 0.74, "std": 0.03, "median": 0.75,
                                  "min": 0.69, "max": 0.79, "num_nulls": 0}
  },
  "per_type": {
    "single_hop": {"count": 15,
                   "final_answer_correctness": {"mean": 0.81, "std": 0.03},
                   "context_recall": {"mean": 0.55, "std": 0.05}}
  }
}
```
