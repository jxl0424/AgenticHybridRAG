"""
Automated hyperparameter tuning for HybridRAG text chunking strategy.

Focuses purely on the `arxiv_papers` collection, measuring chunk limits.
Uses an in-memory ephemeral Qdrant collection to safely re-embed and test chunks 
without touching the persistent Neo4j/Qdrant production data.
"""
import sys
from pathlib import Path
import logging
import optuna
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.local_parquet_loader import LocalParquetLoader, _chunk_md_text
from src.utils import embed_texts_with_model
from src.generation.llm_client import LLMClient
from src.prompts.templates import build_messages
from tests.evaluation.metrics import RAGMetrics

logging.getLogger("rag").setLevel(logging.WARNING)

def get_subset(loader: LocalParquetLoader, domain: str="arxiv_ai", n_papers=5, n_qa=10):
    papers = []
    for i, p in enumerate(loader.iter_papers(domain)):
        papers.append(p)
        if len(papers) >= n_papers: break
        
    qa_pairs = loader.load_local_qa_pairs(domains=[domain], k_per_type=3, stratify=False)
    return papers, qa_pairs[:n_qa]

# Globals to avoid reloading
_loader = LocalParquetLoader()
_papers, _qa_pairs = get_subset(_loader, n_papers=20, n_qa=8)
_llm = LLMClient()
_metrics = RAGMetrics(llm_client=_llm)

def objective(trial):
    # Suggest new chunk sizes
    max_tokens = trial.suggest_int("max_tokens", 200, 1000, step=100)
    min_tokens = trial.suggest_int("min_tokens", 50, max_tokens - 100)
    
    # 1. Ephemeral Qdrant setup
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name="temp_papers",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    # 2. Re-chunk the papers
    points = []
    for paper in _papers:
        if not paper.md_text: continue
        chunks = _chunk_md_text(paper.md_text, min_tokens, max_tokens)
        
        # Embed and formulate payload
        if not chunks: continue
        embeddings = embed_texts_with_model(chunks, "allenai/specter2_base", batch_size=32)
        
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            import uuid
            uid = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=uid,
                    vector=emb,
                    payload={"arxiv_id": paper.arxiv_id, "text": chunk_text}
                )
            )
            
    # Upsert to in-memory Qdrant
    if points:
        for i in range(0, len(points), 256):
            qdrant.upsert("temp_papers", points=points[i:i+256])
            
    # 3. Evaluate QA pairs
    f1_scores = []
    for qa in _qa_pairs:
        question = qa["question"]
        gt_answer = qa["ground_truth_answer"]
        
        # Retrieve context
        q_emb = embed_texts_with_model([question], "allenai/specter2_base", batch_size=1)[0]
        results = qdrant.query_points("temp_papers", query=q_emb, limit=5)
        contexts = [hit.payload["text"] for hit in results.points]
        
        # Generator
        messages = build_messages(question, contexts)
        answer = _llm.generate(messages)
        
        f1 = _metrics.calculate_token_f1(answer, gt_answer)
        f1_scores.append(f1)
        
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return avg_f1


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # Limited trials since LLM is slow
    
    print("\\nBest Chunking Optimization Trial:")
    trial = study.best_trial
    print(f"  Token F1: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
