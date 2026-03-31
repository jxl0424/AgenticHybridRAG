"""
Quick prompt smoke-test against specific questions.

Usage:
    python tests/evaluation/prompt_test.py

Runs questions through the real pipeline (full retrieval + reranking) so prompt
changes can be verified in ~30 seconds without running the full eval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline

QUESTIONS = [
    ("Q2", "If the Expected Free Energy (EFE) objective did NOT mandate minimization of ambiguity, what might be a consequence for an active inference agent?"),
    ("Q8", "If the VC dimension of the class of all complete acyclic 0-bounded CP-nets were not (m-1)n, how might this affect the lower bound on the number of equivalence and membership queries required for learning?"),
]

REFUSAL_PHRASES = ["don't have enough", "cannot answer", "no information"]


def main():
    pipeline = HybridRAGBenchPipeline()

    for label, question in QUESTIONS:
        result = pipeline.query(question, top_k=5, use_hybrid=False)
        answer = result.get("answer", "")
        refused = any(p in answer.lower() for p in REFUSAL_PHRASES)

        print(f"=== {label} ===")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"REFUSED: {refused}")
        print()


if __name__ == "__main__":
    main()
