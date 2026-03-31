"""
Phase 5: Evaluation Report

Loads all result JSON files from tests/results/ and prints a consolidated
comparison table with:
  - Component isolation (vector vs graph vs hybrid vs hybrid+rerank)
  - Reranker delta
  - Agent quality vs generation ceiling (retrieval gap)
  - Category breakdown (factual / relational / procedural)

Usage:
    python tests/evaluation/report.py
    python tests/evaluation/report.py --results-dir tests/results
"""
import json
import os
import sys
import argparse
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "  N/A  "
    return f"{v:.{decimals}f}"


def fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "  N/A  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def print_header(title: str, width: int = 70):
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width)


# ──────────────────────────────────────────────────────────────────────────────
# Section renderers
# ──────────────────────────────────────────────────────────────────────────────

def section_component_isolation(results_dir: str):
    print_header("SECTION 1: COMPONENT ISOLATION (Phase 1)")
    print(f"{'Mode':<28} {'Hit Rate':>10} {'MRR':>10} {'Precision':>10}")
    print("-"*60)

    for label, fname in [
        ("Vector-only",           "vector_only_results.json"),
        ("Graph-only",            "graph_only_results.json"),
        ("Hybrid (no agent)",     "hybrid_results.json"),
    ]:
        data = load(os.path.join(results_dir, fname))
        if data:
            agg = data.get("aggregate", {})
            print(f"{label:<28} {fmt(agg.get('hit_rate')):>10} {fmt(agg.get('mrr')):>10} {fmt(agg.get('context_precision')):>10}")
        else:
            print(f"{label:<28} {'(not run)':>10}")


def section_reranker(results_dir: str):
    print_header("SECTION 2: RERANKER IMPACT (Phase 2)")
    data = load(os.path.join(results_dir, "reranker_impact_results.json"))
    if not data:
        print("  reranker_impact_results.json not found. Run component_eval.py first.")
        return

    before = data.get("before_aggregate", {})
    after = data.get("after_aggregate", {})
    delta = data.get("delta_aggregate", {})

    print(f"{'Stage':<28} {'Hit Rate':>10} {'MRR':>10} {'Precision':>10}")
    print("-"*60)
    print(f"{'Hybrid (before rerank)':<28} {fmt(before.get('hit_rate')):>10} {fmt(before.get('mrr')):>10} {fmt(before.get('context_precision')):>10}")
    print(f"{'Hybrid+Rerank (after)':<28} {fmt(after.get('hit_rate')):>10} {fmt(after.get('mrr')):>10} {fmt(after.get('context_precision')):>10}")
    print(f"{'Delta':<28} {fmt_delta(delta.get('hit_rate')):>10} {fmt_delta(delta.get('mrr')):>10} {fmt_delta(delta.get('context_precision')):>10}")

    hr_delta = delta.get('hit_rate', 0)
    if hr_delta > 0.02:
        print("\n  [ok] Reranker is HELPING (positive delta on hit rate).")
    elif hr_delta < -0.02:
        print("\n  [!] Reranker is HURTING (negative delta on hit rate). Consider disabling it.")
    else:
        print("\n  [~] Reranker has MINIMAL impact.")


def section_agent_vs_ceiling(results_dir: str):
    print_header("SECTION 3: AGENT QUALITY vs GENERATION CEILING (Phase 3 & 4)")

    agent = load(os.path.join(results_dir, "agent_results.json"))
    ceiling = load(os.path.join(results_dir, "generation_ceiling_results.json"))

    print(f"{'Mode':<35} {'Correctness':>12} {'Faithfulness':>13}")
    print("-"*62)

    agent_correct = agent["aggregate_metrics"]["answer_correctness"] if agent else None
    agent_faithful = agent["aggregate_metrics"]["faithfulness"] if agent else None
    ceil_correct = ceiling["aggregate_metrics"]["answer_correctness"] if ceiling else None
    ceil_faithful = ceiling["aggregate_metrics"]["faithfulness"] if ceiling else None

    print(f"{'Agent (full pipeline)':<35} {fmt(agent_correct):>12} {fmt(agent_faithful):>13}")
    print(f"{'Ceiling (GT context -> LLM)':<35} {fmt(ceil_correct):>12} {fmt(ceil_faithful):>13}")

    if agent_correct is not None and ceil_correct is not None:
        gap = ceil_correct - agent_correct
        print(f"\n  Retrieval gap (ceiling - agent correctness): {fmt_delta(gap)} points lost")
        if gap > 0.15:
            print("  [!] RETRIEVAL is the main bottleneck. Improving retrieval will have high impact.")
        elif gap < 0.05:
            print("  [ok] Retrieval is not the bottleneck. LLM quality / prompt design is the limiting factor.")
        else:
            print("  [~] Moderate retrieval loss. Both retrieval and generation improvements are worthwhile.")

    if agent:
        agg = agent["aggregate_metrics"]
        print(f"\n  Agent trajectory averages:")
        print(f"    Steps per query  : {agg.get('trajectory_length', 0):.1f}")
        print(f"    Actions per query: {agg.get('action_count', 0):.1f}")
        print(f"    Tokens per query : {agg.get('total_tokens', 0):.0f}")
        print(f"    Latency (s)      : {agg.get('latency_seconds', 0):.2f}")

        tool_dist = agent.get("tool_usage_distribution", {})
        if tool_dist:
            print("\n  Tool usage distribution:")
            for tool, count in sorted(tool_dist.items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count}")


def section_category_breakdown(results_dir: str):
    print_header("SECTION 4: CATEGORY BREAKDOWN (Phase 3 & 5)")

    agent = load(os.path.join(results_dir, "agent_results.json"))
    ceiling = load(os.path.join(results_dir, "generation_ceiling_results.json"))

    # Collect categories from agent results
    categories = set()
    if agent:
        categories.update(agent.get("category_breakdown", {}).keys())
    if ceiling:
        categories.update(ceiling.get("category_breakdown", {}).keys())

    if not categories:
        print("  No category data available. Run agent_quality_eval.py first.")
        return

    print(f"\n  Agent Quality by Category:")
    print(f"  {'Category':<15} {'N':>5} {'Hit Rate':>10} {'Faithful':>10} {'Correct':>10}")
    print("  " + "-"*52)
    if agent:
        for cat, data in sorted(agent.get("category_breakdown", {}).items()):
            print(f"  {cat:<15} {data['count']:>5} {fmt(data.get('hit_rate')):>10} {fmt(data.get('faithfulness')):>10} {fmt(data.get('answer_correctness')):>10}")

    print(f"\n  Ceiling Quality by Category:")
    print(f"  {'Category':<15} {'N':>5} {'Faithful':>10} {'Correct':>10}")
    print("  " + "-"*43)
    if ceiling:
        for cat, data in sorted(ceiling.get("category_breakdown", {}).items()):
            print(f"  {cat:<15} {data['count']:>5} {fmt(data.get('faithfulness')):>10} {fmt(data.get('answer_correctness')):>10}")

    # Identify worst-performing category
    if agent:
        cat_data = agent.get("category_breakdown", {})
        if cat_data:
            worst = min(cat_data.items(), key=lambda x: x[1].get("answer_correctness", 1.0))
            print(f"\n  [!] Weakest category: '{worst[0]}' (correctness={fmt(worst[1].get('answer_correctness'))})")
            if worst[0] == "relational":
                print("    Graph retrieval may be underperforming for entity-relationship queries.")
            elif worst[0] == "factual":
                print("    Vector search may be missing precise factual chunks.")
            elif worst[0] == "procedural":
                print("    Multi-hop reasoning or answer synthesis may need improvement.")


def section_weaknesses(results_dir: str):
    """Synthesize findings into a prioritized list of weaknesses."""
    print_header("SECTION 5: WEAKNESS DIAGNOSIS & RECOMMENDATIONS")

    vec = load(os.path.join(results_dir, "vector_only_results.json"))
    graph = load(os.path.join(results_dir, "graph_only_results.json"))
    reranker = load(os.path.join(results_dir, "reranker_impact_results.json"))
    agent = load(os.path.join(results_dir, "agent_results.json"))
    ceiling = load(os.path.join(results_dir, "generation_ceiling_results.json"))

    findings = []

    if vec and graph:
        vec_hr = vec["aggregate"].get("hit_rate", 0)
        graph_hr = graph["aggregate"].get("hit_rate", 0)
        if graph_hr < 0.3:
            findings.append(f"[HIGH] Graph retrieval hit rate is low ({graph_hr:.3f}). "
                            "Entity extraction may be failing on queries — check CS entity extractor.")
        if graph_hr < vec_hr * 0.5:
            findings.append(f"[MEDIUM] Graph retrieval ({graph_hr:.3f}) is much weaker than vector "
                            f"({vec_hr:.3f}). Knowledge graph may have sparse coverage.")

    if reranker:
        delta = reranker.get("delta_aggregate", {})
        if delta.get("hit_rate", 0) < -0.02:
            findings.append("[MEDIUM] Reranker reduces hit rate. Cross-encoder may be miscalibrated "
                            "for this domain. Consider disabling or retraining.")

    if agent and ceiling:
        a_corr = agent["aggregate_metrics"].get("answer_correctness", 0)
        c_corr = ceiling["aggregate_metrics"].get("answer_correctness", 0)
        gap = c_corr - a_corr
        if gap > 0.2:
            findings.append(f"[HIGH] Large retrieval gap ({gap:.3f}). The system is losing significant "
                            "quality because the agent is not retrieving the right context.")
        elif gap < 0.05:
            findings.append(f"[LOW] Minimal retrieval gap ({gap:.3f}). Focus on improving the LLM "
                            "prompt or using a more capable model.")

    if agent:
        agg = agent["aggregate_metrics"]
        avg_actions = agg.get("action_count", 0)
        avg_tokens = agg.get("total_tokens", 0)
        if avg_actions > 5:
            findings.append(f"[MEDIUM] Agent uses {avg_actions:.1f} actions per query on average. "
                            "Consider reducing max_steps or simplifying the ReAct prompt.")
        if avg_tokens > 3000:
            findings.append(f"[INFO] High token usage ({avg_tokens:.0f}/query). "
                            "Context window pressure may be degrading later steps.")

    if not findings:
        findings.append("[OK] No critical weaknesses detected.")

    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results_dir: str = "tests/results"):
    print("\n" + "#"*70)
    print("  MEDICAL RAG EVALUATION REPORT")
    print("#"*70)

    section_component_isolation(results_dir)
    section_reranker(results_dir)
    section_agent_vs_ceiling(results_dir)
    section_category_breakdown(results_dir)
    section_weaknesses(results_dir)

    print("\n" + "="*70)
    print("  END OF REPORT")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Report (Phase 5)")
    parser.add_argument("--results-dir", type=str, default="tests/results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()
    generate_report(args.results_dir)
