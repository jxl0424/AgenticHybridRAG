"""
Automated hyperparameter tuning for HybridRAG using Optuna.

Tunes:
- min_score
- chunk_weight
- paper_weight
- graph_weight

Metrics optimized: Mean Reciprocal Rank (MRR)
"""
import sys
import optuna
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
from tests.evaluation.hybridrag_eval import HybridRAGEvaluator
from src.utils import embed_texts_with_model

def objective(trial):
    # Suggest hyperparameters
    min_score = trial.suggest_float("min_score", 0.0, 0.4)
    chunk_weight = trial.suggest_float("chunk_weight", 0.0, 1.0)
    paper_weight = trial.suggest_float("paper_weight", 0.0, 1.0)
    graph_weight = trial.suggest_float("graph_weight", 0.0, 1.0)
    
    # Normalize weights
    total = chunk_weight + paper_weight + graph_weight
    if total == 0:
        total = 1.0
        chunk_weight = paper_weight = graph_weight = 1.0/3.0
    chunk_weight /= total
    paper_weight /= total
    graph_weight /= total

    # Set pipeline configurations safely
    pipeline = HybridRAGBenchPipeline()
    pipeline.min_score = min_score
    pipeline.hybrid_retriever.chunk_weight = chunk_weight
    pipeline.hybrid_retriever.paper_weight = paper_weight
    pipeline.hybrid_retriever.graph_weight = graph_weight
    
    # Disable LLM generator to make this fast - we only care about retrieval metrics!
    evaluator = HybridRAGEvaluator(pipeline)
    qa_pairs = evaluator.load_qa_pairs(k_per_type=10, seed=42) # subset for speed
    
    mrr_scores = []
    
    for qa in qa_pairs:
        question = qa["question"]
        gt_context = qa.get("ground_truth_context", "")
        if not gt_context:
            continue
            
        # Manually run just the retrieval pipeline
        query_embedding = embed_texts_with_model([question], pipeline.embedding_model, batch_size=1)[0]
        
        contexts_raw = pipeline.hybrid_retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=15,
            min_score=min_score,
            use_graph=True,
        )
        
        context_texts = [c.text for c in contexts_raw]
        mrr = evaluator.metrics.calculate_mrr(context_texts, gt_context)
        mrr_scores.append(mrr)
        
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

if __name__ == "__main__":
    import logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("\\nBest trial:")
    trial = study.best_trial
    
    print(f"  Value (MRR): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
