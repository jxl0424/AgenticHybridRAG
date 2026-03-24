"""
Evaluation harness for HybridRAG-Bench.

Loads QA pairs from the HybridRAG-Bench dataset and evaluates
the HybridRAGBenchPipeline across retrieval modes (vector-only vs hybrid).

Usage:
    pipeline = HybridRAGBenchPipeline()
    evaluator = HybridRAGEvaluator(pipeline)
    qa_pairs = evaluator.load_qa_pairs(max_pairs=200)
    results = evaluator.run(qa_pairs, modes=["vector", "hybrid"])
    evaluator.print_comparison_table(results)
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation.llm_client import LLMClient
from src.observability.tracer import start_phoenix, pipeline_span
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
from tests.evaluation.metrics import RAGMetrics


class HybridRAGEvaluator:
    """
    Evaluates HybridRAGBenchPipeline against HybridRAG-Bench QA pairs.

    Compares vector-only vs hybrid retrieval on:
        hit_rate, mrr, context_precision, context_recall,
        ndcg@5, faithfulness, answer_correctness
    """

    def __init__(self, pipeline: HybridRAGBenchPipeline):
        self.pipeline = pipeline
        judge_llm = LLMClient(model="qwen2.5:7b-instruct")
        self.metrics = RAGMetrics(llm_client=judge_llm)
        self.tracer = start_phoenix()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_qa_pairs(
        self,
        split: str = "test",
        max_pairs: Optional[int] = 200,
        local_data_dir: str = "data/hybridrag",
    ) -> list[dict]:
        """
        Load QA pairs from local HybridRAG-Bench parquet files.

        Args:
            split: Dataset split to use ("test" or "train")
            max_pairs: Maximum number of QA pairs to load
            local_data_dir: Path to the local HybridRAG-Bench data directory

        Returns:
            List of dicts with keys: question, ground_truth_answer, ground_truth_context, question_type
        """
        from src.ingestion.local_parquet_loader import LocalParquetLoader
        loader = LocalParquetLoader(local_data_dir)
        return loader.load_local_qa_pairs(split=split, max_pairs=max_pairs)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def run(
        self,
        qa_pairs: list[dict],
        modes: list[str] = None,
        top_k: int = 5,
        output_path: str = "tests/results/hybridrag_eval.json",
    ) -> dict:
        """
        Run evaluation across retrieval modes.

        Args:
            qa_pairs: List of QA dicts from load_qa_pairs()
            modes: Retrieval modes to compare. Options: "vector", "hybrid"
            top_k: Number of contexts to retrieve per query
            output_path: Where to save incremental results

        Returns:
            Aggregated metrics per mode
        """
        if modes is None:
            modes = ["vector", "hybrid"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results: dict[str, list] = {mode: [] for mode in modes}

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            gt_answer = qa.get("ground_truth_answer", "")
            gt_context = qa.get("ground_truth_context", "")

            print(f"[{i+1}/{len(qa_pairs)}] {question}")

            for mode in modes:
                try:
                    with pipeline_span(self.tracer, "query") as root_span:
                        root_span.set_attribute("query.text", question)
                        root_span.set_attribute("retrieval.mode", mode)
                        root_span.set_attribute("retrieval.use_hybrid", mode == "hybrid")
                        root_span.set_attribute("retrieval.top_k", top_k)

                        response = self._run_query(question, mode, top_k)
                        item = self._compute_metrics(question, gt_answer, gt_context, response)
                        item["mode"] = mode

                        m = item.get("metrics", {})
                        for metric_key in ("hit_rate", "mrr", "context_precision", "ndcg_at_5"):
                            val = m.get(metric_key)
                            if isinstance(val, float):
                                root_span.set_attribute(f"eval.{metric_key}", round(val, 4))
                        root_span.set_attribute(
                            "answer.type",
                            response.get("trace", {}).get("answer_type", ""),
                        )
                        root_span.set_attribute(
                            "answer.text", response.get("answer", "")[:500]
                        )

                        results[mode].append(item)
                except Exception as e:
                    print(f"  [{mode}] ERROR: {e}")
                    results[mode].append({
                        "question": question,
                        "mode": mode,
                        "error": str(e),
                        "metrics": {},
                    })

            # Incremental save after every question
            self._save(results, output_path)

        aggregated = self._aggregate(results)
        print(f"\nResults saved to {output_path}")
        return aggregated

    def _run_query(self, question: str, mode: str, top_k: int) -> dict:
        if mode == "hybrid":
            return self.pipeline.query(
                question, top_k=top_k, use_hybrid=True, tracer=self.tracer
            )
        elif mode == "vector":
            return self.pipeline.query(
                question, top_k=top_k, use_hybrid=False, tracer=self.tracer
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_metrics(
        self,
        question: str,
        gt_answer: str,
        gt_context: str,
        response: dict,
    ) -> dict:
        """Compute all retrieval and generation metrics for one QA pair."""
        contexts = response.get("contexts", [])
        answer = response.get("answer", "")

        metrics = {}

        if gt_context:
            metrics["hit_rate"] = self.metrics.calculate_hit_rate(contexts, gt_context)
            metrics["mrr"] = self.metrics.calculate_mrr(contexts, gt_context)
            metrics["context_precision"] = self.metrics.calculate_context_precision(
                question, contexts, gt_context
            )
            metrics["ndcg_at_5"] = self.metrics.calculate_ndcg(
                contexts, [gt_context], k=5
            )

        if gt_answer and contexts:
            metrics["context_recall"] = self.metrics.calculate_context_recall(
                gt_answer, contexts
            )

        if answer and contexts:
            metrics["faithfulness"] = self.metrics.calculate_faithfulness(
                question, contexts, answer
            )

        # TODO(Task4): calculate_answer_correctness now returns dict; this call produces
        # wrong aggregation until _compute_metrics() is replaced in Task 4.
        if gt_answer and answer:
            metrics["answer_correctness"] = self.metrics.calculate_answer_correctness(
                question, gt_answer, answer
            )

        return {
            "question": question,
            "answer": answer,
            "num_contexts": len(contexts),
            "metrics": metrics,
            "trace": response.get("trace", {}),
        }

    def _aggregate(self, results: dict[str, list]) -> dict:
        """Compute mean of each metric per mode."""
        aggregated = {}
        for mode, items in results.items():
            valid = [item for item in items if "metrics" in item and item["metrics"]]
            if not valid:
                aggregated[mode] = {}
                continue

            all_keys = set()
            for item in valid:
                all_keys.update(item["metrics"].keys())

            agg = {}
            for key in all_keys:
                values = [
                    item["metrics"][key]
                    for item in valid
                    if key in item["metrics"] and isinstance(item["metrics"][key], (int, float))
                ]
                agg[key] = round(sum(values) / len(values), 4) if values else None

            agg["num_evaluated"] = len(valid)
            aggregated[mode] = agg

        return aggregated

    def _save(self, results: dict, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def print_trace_summary(self, results: dict) -> None:
        """Print a per-question breakdown of pipeline internals for failed/interesting cases."""
        print("\n" + "=" * 90)
        print("Pipeline Trace Summary")
        print("=" * 90)
        header = f"{'#':<3} {'mode':<8} {'ans':<8} {'entities':<8} {'graph_ids':<10} {'pre_rr':<7} {'post_rr':<8} {'dropped':<8} {'cr':<6} {'faith':<6} {'corr':<6} question"
        print(header)
        print("-" * 90)

        for mode, items in results.items():
            for i, item in enumerate(items):
                t = item.get("trace", {})
                m = item.get("metrics", {})
                ans_type = t.get("answer_type", "?")[:7]
                n_entities = len(t.get("entities_extracted", []))
                graph_ids = t.get("graph_qdrant_ids_total", 0)
                pre_rr = t.get("pre_rerank_count", 0)
                post_rr = t.get("post_rerank_count", 0)
                dropped = t.get("dropped_by_reranker", 0)
                cr = m.get("context_recall", "")
                faith = m.get("faithfulness", "")
                corr = m.get("answer_correctness", "")
                q = item.get("question", "")[:45]
                cr_s = f"{cr:.2f}" if isinstance(cr, float) else "N/A"
                fa_s = f"{faith:.2f}" if isinstance(faith, float) else "N/A"
                co_s = f"{corr:.2f}" if isinstance(corr, float) else "N/A"
                print(f"{i+1:<3} {mode:<8} {ans_type:<8} {n_entities:<8} {graph_ids:<10} {pre_rr:<7} {post_rr:<8} {dropped:<8} {cr_s:<6} {fa_s:<6} {co_s:<6} {q}")

        print("=" * 90)
        print("Columns: ans=answer_type, entities=# extracted, graph_ids=qdrant_ids resolved,")
        print("         pre_rr=contexts before rerank, post_rr=after, dropped=filtered by threshold")
        print()

    def print_comparison_table(self, aggregated: dict) -> None:
        """Print a side-by-side comparison table of modes."""
        metric_keys = [
            "hit_rate", "mrr", "context_precision", "context_recall",
            "ndcg_at_5", "faithfulness", "answer_correctness",
        ]

        modes = list(aggregated.keys())
        col_w = 18

        header = f"{'Metric':<24}" + "".join(f"{m:>{col_w}}" for m in modes)
        print("\n" + "=" * len(header))
        print("HybridRAG-Bench Evaluation Results")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for key in metric_keys:
            row = f"{key:<24}"
            for mode in modes:
                val = aggregated.get(mode, {}).get(key)
                if val is None:
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{val:>{col_w}.4f}"
            print(row)

        print("-" * len(header))
        n_row = f"{'num_evaluated':<24}"
        for mode in modes:
            n = aggregated.get(mode, {}).get("num_evaluated", 0)
            n_row += f"{n:>{col_w}}"
        print(n_row)
        print("=" * len(header) + "\n")


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HybridRAG-Bench pipeline")
    parser.add_argument("--max-pairs", type=int, default=50, help="Number of QA pairs to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Contexts per query")
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid"], help="Retrieval modes")
    parser.add_argument("--output", default="tests/results/hybridrag_eval.json", help="Output path")
    parser.add_argument("--local-data-dir", default="data/hybridrag", help="Path to local HybridRAG-Bench parquet data")
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipeline = HybridRAGBenchPipeline()
    evaluator = HybridRAGEvaluator(pipeline)

    print(f"Loading {args.max_pairs} QA pairs from {args.local_data_dir}...")
    qa_pairs = evaluator.load_qa_pairs(max_pairs=args.max_pairs, local_data_dir=args.local_data_dir)

    if not qa_pairs:
        print("No QA pairs found. Inspect the dataset schema printed above and adjust --question-key etc.")
        sys.exit(1)

    print(f"Running evaluation: modes={args.modes}, top_k={args.top_k}")
    aggregated = evaluator.run(qa_pairs, modes=args.modes, top_k=args.top_k, output_path=args.output)
    evaluator.print_comparison_table(aggregated)

    import json
    with open(args.output) as f:
        raw_results = json.load(f)
    evaluator.print_trace_summary(raw_results)

    if evaluator.tracer is not None:
        print("\nPhoenix is running at http://localhost:6006")
        print("Press Enter to shut down Phoenix and exit...")
        input()
