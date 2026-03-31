"""
HybridRAG-Bench pipeline for CS/AI domain ingestion and querying.

Wires together:
- SPECTER2 embeddings (allenai/specter2_base)
- CSEntityExtractor (CS/AI NER)
- CSKnowledgeGraph (Neo4j :CSEntity label)
- ChunkRetriever (arxiv_chunks collection)
- PaperRetriever (arxiv_papers collection)
- HybridRetriever (RRF fusion)
- Reranker + LLMClient (generation)
"""
import json
import os
import time
import yaml
from pathlib import Path
from typing import Optional

from src.utils import embed_texts_with_model
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.paper_retriever import PaperRetriever
from src.graph.cs_entity_extractor import CSEntityExtractor, get_cs_entity_extractor
from src.graph.cs_knowledge_graph import CSKnowledgeGraph
from src.graph.cs_knowledge_graph import GraphDocument, GraphChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm_client import LLMClient, build_providers_from_config
from src.prompts.templates import build_messages
from src.utils import get_logger

logger = get_logger(__name__)


class _ProgressTracker:
    """Minimal progress tracker backed by a JSON file."""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self._ingested: set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                self._ingested = set(data.get("ingested", []))
                logger.info(f"Resumed progress: {len(self._ingested)} records already ingested")
            except Exception:
                pass

    def _save(self):
        Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump({"ingested": list(self._ingested)}, f)

    def is_ingested(self, record_id: str) -> bool:
        return record_id in self._ingested

    def mark_ingested(self, record_id: str):
        self._ingested.add(record_id)

    def flush(self):
        self._save()


class HybridRAGBenchPipeline:
    """
    End-to-end pipeline for the HybridRAG-Bench (arXiv CS/AI) domain.

    Usage:
        pipeline = HybridRAGBenchPipeline()
        result = pipeline.query("What is the attention mechanism in transformers?")
    """

    def __init__(self, config_path: str = "config/defaults.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        hcfg = cfg.get("hybridrag", {})
        hybrid_cfg = cfg.get("hybrid_retrieval", {})

        self.embedding_model: str = hcfg.get("embedding_model", "allenai/specter2_base")
        self.embedding_dim: int = hcfg.get("embedding_dim", 768)
        self.batch_size: int = hcfg.get("batch_size", 64)
        self.progress_file: str = hcfg.get("progress_file", "data/hybridrag/.ingest_progress.json")

        # Vector retrievers
        self.chunk_retriever = ChunkRetriever()
        self.paper_retriever = PaperRetriever()

        # Knowledge graph
        self.cs_kg = CSKnowledgeGraph()
        self.cs_kg.initialize_schema()

        # Entity extractor (CS/AI domain)
        self.entity_extractor: CSEntityExtractor = get_cs_entity_extractor()

        # Graph retriever wired to the CS KG and CS extractor
        self.graph_retriever = GraphRetriever(
            knowledge_graph=self.cs_kg,
            entity_extractor=self.entity_extractor,  # duck-typed
            chunk_retriever=self.chunk_retriever,
        )

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            chunk_retriever=self.chunk_retriever,
            paper_retriever=self.paper_retriever,
            graph_retriever=self.graph_retriever,
        )

        # Reranker
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        self.reranker = Reranker(model_name="msmarco", device=device)

        # LLM — OpenRouter first, Ollama fallback (see config/defaults.yaml openrouter block)
        self.llm = LLMClient(providers=build_providers_from_config(cfg))

        self.min_score: float = hcfg.get("min_score", hybrid_cfg.get("min_score", 0.05))

    # -------------------------------------------------------------------------
    # Native KG ingestion
    # -------------------------------------------------------------------------

    def ingest_native_kg(self, batch_size: int = 500) -> dict:
        """
        Import the dataset's pre-built knowledge graph directly into Neo4j.

        Loads nodes.parquet and edges.parquet from all three subsets
        (arxiv_ai, arxiv_cy, arxiv_qm) and writes them as CSEntity nodes
        and typed relationships, bypassing the rule-based CSEntityExtractor.

        Args:
            batch_size: Neo4j write batch size for nodes and edges

        Returns:
            Dict with nodes_imported and edges_imported counts
        """
        from src.ingestion.hf_hybridrag_loader import HybridRAGKGLoader  # noqa: PLC0415
        logger.info("Loading native KG from HybridRAG-Bench dataset...")
        kg_loader = HybridRAGKGLoader()

        # --- Nodes ---
        nodes_df = kg_loader.load_nodes()
        logger.info(f"Loaded {len(nodes_df)} entity nodes from dataset")

        node_records = (
            nodes_df[["node_id", "display_name", "primary_label"]]
            .drop_duplicates(subset=["node_id"])
            .to_dict("records")
        )
        total_nodes = 0
        for i in range(0, len(node_records), batch_size):
            batch = node_records[i : i + batch_size]
            self.cs_kg.client.execute_write(
                """
                UNWIND $batch AS n
                MERGE (e:CSEntity {name: n.display_name})
                SET e.node_id   = n.node_id,
                    e.entity_type = n.primary_label
                """,
                {"batch": batch},
            )
            total_nodes += len(batch)
        logger.info(f"Imported {total_nodes} CSEntity nodes")

        # --- Edges ---
        edges_df = kg_loader.load_edges()
        node_map = dict(zip(nodes_df["node_id"], nodes_df["display_name"]))
        edges_df["src_name"] = edges_df["src_id"].map(node_map)
        edges_df["dst_name"] = edges_df["dst_id"].map(node_map)
        edges_df = edges_df.dropna(subset=["src_name", "dst_name"])
        logger.info(f"Loaded {len(edges_df)} edges from dataset")

        total_edges = 0
        for rel_type, group in edges_df.groupby("rel_type"):
            clean_rel = (
                "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
                or "RELATED_TO"
            )
            records = (
                group[["src_name", "dst_name"]]
                .rename(columns={"src_name": "source", "dst_name": "target"})
                .to_dict("records")
            )
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.cs_kg.client.execute_write(
                    f"""
                    UNWIND $batch AS rel
                    MATCH (e1:CSEntity {{name: rel.source}})
                    MATCH (e2:CSEntity {{name: rel.target}})
                    MERGE (e1)-[r:{clean_rel}]->(e2)
                    """,
                    {"batch": batch},
                )
            total_edges += len(records)

        logger.info(f"Imported {total_edges} relationships")
        return {"nodes_imported": total_nodes, "edges_imported": total_edges}

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5, use_hybrid: bool = True, tracer=None) -> dict:
        """
        Query the CS/AI RAG system.

        Args:
            question: Natural language question
            top_k: Number of final contexts to return
            use_hybrid: If True, use vector + graph hybrid; otherwise vector only
            tracer: Optional OTel tracer from start_phoenix(). When None all span
                    calls are no-ops and behaviour is identical to the untraced path.

        Returns:
            Dict with answer, contexts, sources, entities_found, retrieval_type, trace
        """
        from src.observability.tracer import pipeline_span

        # --- Embed ---
        with pipeline_span(tracer, "embed_query") as span:
            t0 = time.perf_counter()
            query_embedding = embed_texts_with_model(
                [question], self.embedding_model, batch_size=1
            )[0]
            span.set_attribute("embedding.model", self.embedding_model)
            span.set_attribute("embedding.vector_dim", len(query_embedding))
            span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

        # --- Retrieve (all three sources, timed as a unit) ---
        t0 = time.perf_counter()
        contexts_raw = self.hybrid_retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            min_score=self.min_score,
            use_graph=use_hybrid,
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Only read graph trace when graph was actually called — otherwise _last_trace
        # holds stale data from the previous question's hybrid run.
        _empty_graph_trace = {"entities_extracted": [], "qdrant_ids_per_entity": {}, "total_qdrant_ids": 0, "fetched_count": 0}
        graph_trace = getattr(self.graph_retriever, "_last_trace", {}) if use_hybrid else _empty_graph_trace
        hybrid_trace = getattr(self.hybrid_retriever, "_last_trace", {})

        # Sub-spans from trace data — attribute-only, timing is the shared retrieve_ms
        with pipeline_span(tracer, "chunk_retriever") as span:
            span.set_attribute("chunk.raw_count", hybrid_trace.get("chunk_count_raw", 0))
            span.set_attribute("latency_ms", retrieve_ms)

        with pipeline_span(tracer, "paper_retriever") as span:
            span.set_attribute("paper.raw_count", hybrid_trace.get("paper_count_raw", 0))
            span.set_attribute("latency_ms", retrieve_ms)

        with pipeline_span(tracer, "graph_retriever") as span:
            entities = graph_trace.get("entities_extracted", [])
            span.set_attribute("graph.entity_count", len(entities))
            span.set_attribute("graph.total_qdrant_ids", graph_trace.get("total_qdrant_ids", 0))
            span.set_attribute("graph.fetched_count", graph_trace.get("fetched_count", 0))
            span.set_attribute("latency_ms", retrieve_ms)

            with pipeline_span(tracer, "entity_extraction") as s:
                s.set_attribute("entities_extracted", json.dumps(
                    graph_trace.get("entities_extracted", [])
                ))
                s.set_attribute("latency_ms", retrieve_ms)

            with pipeline_span(tracer, "chunk_fetch") as s:
                s.set_attribute("ids_requested", graph_trace.get("total_qdrant_ids", 0))
                s.set_attribute("ids_resolved", graph_trace.get("fetched_count", 0))
                s.set_attribute("latency_ms", retrieve_ms)

        with pipeline_span(tracer, "rrf_fuse") as span:
            span.set_attribute("fused.count", hybrid_trace.get("fused_count", 0))
            span.set_attribute("fused.source_breakdown", json.dumps(
                hybrid_trace.get("fused_source_breakdown", {})
            ))
            span.set_attribute("top_rrf_scores", json.dumps(
                hybrid_trace.get("top_rrf_scores", [])
            ))
            span.set_attribute("rrf.weights", json.dumps({
                "chunk_weight": self.hybrid_retriever.chunk_weight,
                "paper_weight": self.hybrid_retriever.paper_weight,
                "graph_weight": self.hybrid_retriever.graph_weight,
            }))

        context_texts = [c.text for c in contexts_raw]
        context_sources = [c.source for c in contexts_raw]
        context_scores = [c.score for c in contexts_raw]
        entities_found = list({
            e
            for c in contexts_raw
            for e in c.metadata.get("entities_found", [])
        })

        # Snapshot before reranking
        pre_rerank_count = len(context_texts)
        pre_rerank_source_counts: dict[str, int] = {}
        for c in contexts_raw:
            pre_rerank_source_counts[c.collection] = (
                pre_rerank_source_counts.get(c.collection, 0) + 1
            )

        # --- Rerank (two stages: cross-encoder + relative threshold) ---
        if self.reranker and context_texts:
            with pipeline_span(tracer, "rerank_crossencoder") as span:
                t0 = time.perf_counter()
                reranked = self.reranker.rerank(
                    query=question,
                    documents=context_texts,
                    sources=context_sources,
                    scores=context_scores,
                    top_k=top_k,
                )
                context_texts = reranked.get("contexts", context_texts[:top_k])
                context_sources = reranked.get("sources", context_sources[:top_k])
                context_scores = reranked.get("rerank_scores", context_scores[:top_k])
                span.set_attribute("pre_count", pre_rerank_count)
                span.set_attribute("post_count", len(context_texts))
                span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

            pre_threshold_count = len(context_texts)
            top_score = max(context_scores) if context_scores else 0
            keep = [i for i, s in enumerate(context_scores) if s >= top_score - 8]

            with pipeline_span(tracer, "rerank_threshold") as span:
                span.set_attribute("pre_count", pre_threshold_count)
                span.set_attribute("threshold_dropped", pre_threshold_count - len(keep))
                span.set_attribute("post_count", len(keep))

            context_texts = [context_texts[i] for i in keep]
            context_sources = [context_sources[i] for i in keep]
            context_scores = [context_scores[i] for i in keep]
        else:
            context_texts = context_texts[:top_k]
            context_sources = context_sources[:top_k]
            context_scores = context_scores[:top_k]

        # --- Generate ---
        messages = build_messages(question, context_texts)

        with pipeline_span(tracer, "llm_generate") as span:
            t0 = time.perf_counter()
            answer = self.llm.generate(messages)
            span.set_attribute("llm.model", self.llm._active_model)
            span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

        _refusal_phrases = [
            "i don't have enough information",
            "i do not have enough information",
            "i don't know",
            "cannot answer",
            "no information",
        ]
        answer_type = (
            "refusal" if any(p in answer.lower() for p in _refusal_phrases) else "answer"
        )

        trace = {
            "entities_extracted": graph_trace.get("entities_extracted", []),
            "qdrant_ids_per_entity": graph_trace.get("qdrant_ids_per_entity", {}),
            "graph_qdrant_ids_total": graph_trace.get("total_qdrant_ids", 0),
            "graph_fetched_count": graph_trace.get("fetched_count", 0),
            "raw_counts": {
                "chunk": hybrid_trace.get("chunk_count_raw", 0),
                "paper": hybrid_trace.get("paper_count_raw", 0),
                "graph": hybrid_trace.get("graph_count_raw", 0),
            },
            "pre_rerank_count": pre_rerank_count,
            "pre_rerank_source_breakdown": pre_rerank_source_counts,
            "post_rerank_count": len(context_texts),
            "dropped_by_reranker": pre_rerank_count - len(context_texts),
            "reranker_scores": [round(s, 3) for s in context_scores],
            "answer_type": answer_type,
            "top3_contexts": [
                {"source": s, "rerank_score": round(sc, 3), "text": t[:150]}
                for t, s, sc in zip(
                    context_texts[:3], context_sources[:3], context_scores[:3]
                )
            ],
        }

        return {
            "question": question,
            "answer": answer,
            "contexts": context_texts,
            "sources": context_sources,
            "scores": context_scores,
            "entities_found": entities_found,
            "retrieval_type": "hybrid" if use_hybrid else "vector",
            "trace": trace,
        }


# Module-level singleton
_default_pipeline: Optional[HybridRAGBenchPipeline] = None


def get_pipeline(config_path: str = "config/defaults.yaml") -> HybridRAGBenchPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = HybridRAGBenchPipeline(config_path)
    return _default_pipeline
