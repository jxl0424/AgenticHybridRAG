"""
Full pipeline query test for HybridRAG.

Usage:
    python query_hybridrag.py "your question here"
    python query_hybridrag.py  # uses default question
    python query_hybridrag.py "your question" --vector-only
    python query_hybridrag.py "your question" --top-k 10
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.CRITICAL)

SEPARATOR = "=" * 60


def parse_args():
    parser = argparse.ArgumentParser(description="Query the HybridRAG pipeline")
    parser.add_argument("question", nargs="?", default="what is predictive coding in neural networks")
    parser.add_argument("--vector-only", action="store_true", help="Skip graph retrieval, use vector search only")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final contexts (default: 5)")
    parser.add_argument("--min-rerank-score", type=float, default=None, help="Drop contexts with reranker score below this (e.g. 0.0 drops all negative-scored results)")
    parser.add_argument("--compare", action="store_true", help="Run both hybrid and vector-only, print side by side")
    return parser.parse_args()


def run_query(pipeline, question: str, top_k: int, use_hybrid: bool) -> dict:
    return pipeline.query(question, top_k=top_k, use_hybrid=use_hybrid)


def filter_by_rerank_score(result: dict, min_score: float) -> dict:
    """Remove contexts whose reranker score is below min_score."""
    keep = [i for i, s in enumerate(result['scores']) if s >= min_score]
    result['contexts'] = [result['contexts'][i] for i in keep]
    result['sources']  = [result['sources'][i]  for i in keep]
    result['scores']   = [result['scores'][i]   for i in keep]
    return result


def print_result(result: dict, label: str):
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    print(f"  Question : {result['question']}")
    print(f"  Mode     : {result['retrieval_type']}")
    print(f"  Contexts : {len(result['contexts'])}")
    if result['entities_found']:
        print(f"  Entities : {result['entities_found']}")
    print()
    print("  ANSWER")
    print("  " + "-" * 56)
    for line in result['answer'].splitlines():
        print(f"  {line}")
    print()
    print("  RETRIEVED CONTEXTS")
    print("  " + "-" * 56)
    for i, (ctx, score, source) in enumerate(
        zip(result['contexts'], result['scores'], result['sources']), 1
    ):
        print(f"  [{i}] score={score:.4f}  source={source or 'n/a'}")
        print(f"       {ctx[:200].strip()}")
        print()


def main():
    args = parse_args()

    print(SEPARATOR)
    print("  Initializing HybridRAG pipeline...")
    print(SEPARATOR)

    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
    pipeline = HybridRAGBenchPipeline()

    print(f"  Ready. Querying: {args.question!r}\n")

    if args.compare:
        hybrid = run_query(pipeline, args.question, args.top_k, use_hybrid=True)
        vector = run_query(pipeline, args.question, args.top_k, use_hybrid=False)
        if args.min_rerank_score is not None:
            hybrid = filter_by_rerank_score(hybrid, args.min_rerank_score)
            vector = filter_by_rerank_score(vector, args.min_rerank_score)
        print_result(hybrid, "HYBRID (vector + graph)")
        print_result(vector, "VECTOR ONLY")
    else:
        result = run_query(pipeline, args.question, args.top_k, use_hybrid=not args.vector_only)
        if args.min_rerank_score is not None:
            result = filter_by_rerank_score(result, args.min_rerank_score)
        print_result(result, "HYBRID" if not args.vector_only else "VECTOR ONLY")

    print(SEPARATOR)


if __name__ == "__main__":
    main()
