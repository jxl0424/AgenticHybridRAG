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
from src.generation.llm_client import LLMClient
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
        llm_cfg = cfg.get("llm", {})
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

        # LLM
        self.llm = LLMClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "qwen2.5:7b-instruct"),
        )

        self.min_score: float = hcfg.get("min_score", hybrid_cfg.get("min_score", 0.05))

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    def ingest(self, *args, **kwargs):
        raise NotImplementedError(
            "HybridRAGBenchPipeline.ingest() is deprecated. "
            "Use ingest_local.py (LocalIngestionPipeline) instead."
        )

    def _flush_batch(
        self,
        neo4j_batch: list,
        qdrant_ids: list[int],
        qdrant_payloads: list[dict],
    ) -> None:
        """Store one batch in Qdrant + Neo4j.

        Uses the pre-computed embedding stored in each batch item's third element
        when available, falling back to on-the-fly SPECTER2 encoding only for
        chunks that are missing it.
        """
        precomputed = [item[2] if len(item) > 2 else None for item in neo4j_batch]

        if all(v is not None for v in precomputed):
            vectors = precomputed
        else:
            # Some or all embeddings missing — compute for those that need it
            texts = [item[0].text for item in neo4j_batch]
            computed = embed_texts_with_model(texts, self.embedding_model, self.batch_size)
            vectors = [
                pre if pre is not None else comp
                for pre, comp in zip(precomputed, computed)
            ]

        # Strip the embedding element before passing to Neo4j
        neo4j_pairs = [(item[0], item[1]) for item in neo4j_batch]
        self.cs_kg.ingest_extraction_results_batch(neo4j_pairs)

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

    def query(self, question: str, top_k: int = 5, use_hybrid: bool = True) -> dict:
        """
        Query the CS/AI RAG system.

        Args:
            question: Natural language question
            top_k: Number of final contexts to return
            use_hybrid: If True, use vector + graph hybrid; otherwise vector only

        Returns:
            Dict with answer, contexts, sources, entities_found, retrieval_type
        """
        query_embedding = embed_texts_with_model(
            [question], self.embedding_model, batch_size=1
        )[0]

        contexts_raw = self.hybrid_retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            min_score=self.min_score,
            use_graph=use_hybrid,
        )

        context_texts = [c.text for c in contexts_raw]
        context_sources = [c.source for c in contexts_raw]
        context_scores = [c.score for c in contexts_raw]
        entities_found = list({
            e
            for c in contexts_raw
            for e in c.metadata.get("entities_found", [])
        })

        # Rerank
        if self.reranker and context_texts:
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

            # Drop contexts the reranker scores as irrelevant (negative cross-encoder logits)
            keep = [i for i, s in enumerate(context_scores) if s >= 0]
            if keep:
                context_texts = [context_texts[i] for i in keep]
                context_sources = [context_sources[i] for i in keep]
                context_scores = [context_scores[i] for i in keep]
        else:
            context_texts = context_texts[:top_k]
            context_sources = context_sources[:top_k]
            context_scores = context_scores[:top_k]

        messages = build_messages(question, context_texts)
        answer = self.llm.generate(messages)

        return {
            "question": question,
            "answer": answer,
            "contexts": context_texts,
            "sources": context_sources,
            "scores": context_scores,
            "entities_found": entities_found,
            "retrieval_type": "hybrid" if use_hybrid else "vector",
        }


# Module-level singleton
_default_pipeline: Optional[HybridRAGBenchPipeline] = None


def get_pipeline(config_path: str = "config/defaults.yaml") -> HybridRAGBenchPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = HybridRAGBenchPipeline(config_path)
    return _default_pipeline
