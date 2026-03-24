"""
Evaluation harness for HybridRAG-Bench.

Loads QA pairs from the HybridRAG-Bench dataset and evaluates
the HybridRAGBenchPipeline across retrieval modes (vector-only vs hybrid).

Usage:
    pipeline = HybridRAGBenchPipeline()
    evaluator = HybridRAGEvaluator(pipeline)
    qa_pairs = evaluator.load_qa_pairs(k_per_type=5, seed=42)
    results = evaluator.run(qa_pairs, modes=["vector", "hybrid"])
    evaluator.print_comparison_table(results)
"""
import json
import os
import statistics
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation.llm_client import LLMClient
from src.observability.tracer import start_phoenix, pipeline_span
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
from tests.evaluation.metrics import RAGMetrics


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def validate_output_path(path: str) -> None:
    """Raise ValueError if path does not end in .json."""
    if not path.endswith(".json"):
        raise ValueError(f"--output must end in .json, got: {path!r}")


def _derive_run_path(output: str, run_index: int) -> str:
    """'my_eval.json' -> 'my_eval_run1.json'"""
    p = Path(output)
    return (p.parent / f"{p.stem}_run{run_index}{p.suffix}").as_posix()


def _derive_summary_path(output: str) -> str:
    """'my_eval.json' -> 'my_eval_summary.json'"""
    p = Path(output)
    return (p.parent / f"{p.stem}_summary{p.suffix}").as_posix()


class HybridRAGEvaluator:
    """
    Evaluates HybridRAGBenchPipeline against HybridRAG-Bench QA pairs.

    Compares vector-only vs hybrid retrieval on:
        exact_match, token_f1, answer_correctness_llm, final_answer_correctness,
        context_recall, faithfulness (+ hit_rate, mrr, context_precision, ndcg_at_5 when gt_context available)
    """

    EXTRACTIVE_TYPES = frozenset({"single_hop", "single_hop_w_conditions", "multi_hop"})
    GENERATIVE_TYPES = frozenset({"open_ended", "counterfactual"})

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
        local_data_dir: str = "data/hybridrag",
        seed: int | None = None,
        k_per_type: int = 5,
        stratify: bool = True,
    ) -> list[dict]:
        """
        Load QA pairs from local HybridRAG-Bench parquet files using stratified sampling.

        Args:
            split: Dataset split to use ("test" or "train")
            local_data_dir: Path to the local HybridRAG-Bench data directory
            seed: RNG seed for reproducible sampling (None = random)
            k_per_type: Questions sampled per question_type
            stratify: Whether to stratify by question_type

        Returns:
            List of dicts with keys: question, ground_truth_answer, ground_truth_context,
            question_type, domain
        """
        from src.ingestion.local_parquet_loader import LocalParquetLoader
        loader = LocalParquetLoader(local_data_dir)
        return loader.load_local_qa_pairs(
            split=split, seed=seed, k_per_type=k_per_type, stratify=stratify
        )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def run(
        self,
        qa_pairs: list[dict],
        modes: list[str] = None,
        top_k: int = 5,
        output_path: str = "tests/results/hybridrag_eval.json",
        seed: int = 0,
        run: int = 1,
    ) -> dict:
        """
        Run evaluation across retrieval modes.

        Args:
            qa_pairs: List of QA dicts from load_qa_pairs()
            modes: Retrieval modes to compare. Options: "vector", "hybrid"
            top_k: Number of contexts to retrieve per query
            output_path: Where to save incremental results
            seed: RNG seed used for this run (stored in each per-question item)
            run: Run index (1-based, stored in each per-question item)

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
                        item = self._compute_metrics(
                            question, gt_answer, gt_context, response,
                            question_type=qa.get("question_type", ""),
                            domain=qa.get("domain", ""),
                            seed=seed,
                            run=run,
                        )
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
                if tf1 == 0.0 and pred_len > 15:  # long prediction with zero F1 -> likely verbose, not simply wrong
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
                corr = m.get("final_answer_correctness", "")
                q = item.get("question", "")[:45]
                cr_s = f"{cr:.2f}" if isinstance(cr, float) else "N/A"
                fa_s = f"{faith:.2f}" if isinstance(faith, float) else "N/A"
                co_s = f"{corr:.2f}" if isinstance(corr, float) else "N/A"
                print(f"{i+1:<3} {mode:<8} {ans_type:<8} {n_entities:<8} {graph_ids:<10} {pre_rr:<7} {post_rr:<8} {dropped:<8} {cr_s:<6} {fa_s:<6} {co_s:<6} {q}")

        print("=" * 90)
        print("Columns: ans=answer_type, entities=# extracted, graph_ids=qdrant_ids resolved,")
        print("         pre_rr=contexts before rerank, post_rr=after, dropped=filtered by threshold")
        print()

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
                values = [m.get(key) for m in metric_list if isinstance(m.get(key), float)]
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
        def _fmt(val):
            return f"{val:.4f}" if isinstance(val, float) else "N/A"

        print("\n" + "=" * 80)
        print("Breakdown by Question Type")
        print("=" * 80)

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

    def print_comparison_table(self, aggregated: dict) -> None:
        """Print a side-by-side comparison table of modes."""
        metric_keys = [
            "exact_match", "token_f1", "answer_correctness_llm",
            "final_answer_correctness", "context_recall", "faithfulness",
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
