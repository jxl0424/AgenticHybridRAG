"""
Local ingestion pipeline for HybridRAG-Bench parquet data.

Orchestrates:
  - Startup validation and connectivity checks
  - Optional --reset (wipe Neo4j + Qdrant + progress file)
  - Index / collection creation
  - Per-domain ingestion in 4 phases:
      1. nodes + paper nodes
      2. edges
      3. chunks (arxiv_chunks Qdrant)
      4. paper chunks (arxiv_papers Qdrant via SPECTER2)
  - Resume via progress file
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from neo4j import GraphDatabase

from src.ingestion.local_parquet_loader import (
    LocalParquetLoader,
    DOMAINS,
    NodeRecord,
    EdgeRecord,
    ChunkRecord,
    PaperChunkRecord,
)
from src.utils import embed_texts_with_model
from src.utils import get_logger

logger = get_logger(__name__)

PROGRESS_FILE = "data/hybridrag/.local_ingest_progress.json"
EMBEDDING_MODEL = "allenai/specter2_base"
EMBEDDING_DIM = 768
NEO4J_BATCH = 500
QDRANT_BATCH = 256

COLLECTIONS = {
    "arxiv_nodes":  {"size": EMBEDDING_DIM, "distance": Distance.COSINE},
    "arxiv_chunks": {"size": EMBEDDING_DIM, "distance": Distance.COSINE},
    "arxiv_papers": {"size": EMBEDDING_DIM, "distance": Distance.COSINE},
}


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, path: str):
        self.path = Path(path)
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, domain: str, phase: str) -> bool:
        return self._data.get(domain, {}).get(phase, False)

    def mark_done(self, domain: str, phase: str):
        self._data.setdefault(domain, {})[phase] = True
        self._save()

    def mark_failed(self, domain: str, phase: str):
        self._data.setdefault(domain, {})[phase] = False
        self._save()

    def log_batch_failure(self, domain: str, phase: str, batch_index: int, error: str, record_ids: list):
        entry = {
            "domain": domain,
            "phase": phase,
            "batch_index": batch_index,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_ids": record_ids,
        }
        self._data.setdefault("failed_batches", []).append(entry)
        self._save()

    def delete(self):
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class LocalIngestionPipeline:
    """
    Ingests HybridRAG-Bench from local parquet files into Neo4j + Qdrant.

    Usage:
        pipeline = LocalIngestionPipeline()
        pipeline.run(reset=True, domains=["arxiv_ai"])
    """

    def __init__(
        self,
        data_dir: str = "data/hybridrag",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Pokemon0424$",
        qdrant_url: str = "http://localhost:6333",
        neo4j_batch_size: int = NEO4J_BATCH,
        progress_file: str = PROGRESS_FILE,
    ):
        self.loader = LocalParquetLoader(data_dir)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_url = qdrant_url
        self.neo4j_batch_size = neo4j_batch_size
        self.progress_file = progress_file

        self._driver = None
        self._qdrant: Optional[QdrantClient] = None
        self._progress: Optional[_Progress] = None

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------

    def _neo4j(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
        return self._driver

    def _q(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url, timeout=60)
        return self._qdrant

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _ping_neo4j(self):
        try:
            self._neo4j().verify_connectivity()
            logger.info("Neo4j reachable")
        except Exception as e:
            raise RuntimeError(f"Neo4j unreachable: {e}")

    def _ping_qdrant(self):
        try:
            self._q().get_collections()
            logger.info("Qdrant reachable")
        except Exception as e:
            raise RuntimeError(f"Qdrant unreachable: {e}")

    def _check_sentence_transformers(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "sentence_transformers not installed. Run: pip install sentence-transformers"
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self):
        logger.info("RESET: wiping Neo4j...")
        with self._neo4j().session() as s:
            s.run("MATCH (n) DETACH DELETE n")

        logger.info("RESET: dropping and recreating Qdrant collections...")
        q = self._q()
        for name, cfg in COLLECTIONS.items():
            if q.collection_exists(name):
                q.delete_collection(name)
            q.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=cfg["size"], distance=cfg["distance"]),
            )

        _Progress(self.progress_file).delete()
        logger.info("RESET complete")

    # ------------------------------------------------------------------
    # Index / collection setup
    # ------------------------------------------------------------------

    def _ensure_collections(self):
        q = self._q()
        for name, cfg in COLLECTIONS.items():
            if not q.collection_exists(name):
                q.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=cfg["size"], distance=cfg["distance"]),
                )
                logger.info(f"Created Qdrant collection: {name}")

    def _create_neo4j_indexes(self):
        queries = [
            "CREATE INDEX embeddable_node_domain IF NOT EXISTS FOR (n:_Embeddable) ON (n.node_id, n.domain)",
            "CREATE INDEX paper_arxiv_id IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)",
        ]
        with self._neo4j().session() as s:
            for q in queries:
                s.run(q)
        logger.info("Neo4j indexes created (nodes + papers)")

    def _create_rel_indexes(self, domain: str):
        """Create per-rel_type edge_id indexes after edges phase."""
        import pandas as pd
        path = self.loader.data_dir / f"kg/{domain}/edges.parquet"
        df = pd.read_parquet(path, columns=["rel_type"])
        rel_types = df["rel_type"].dropna().unique()
        with self._neo4j().session() as s:
            for rt in rel_types:
                clean = "".join(c for c in rt if c.isalnum() or c == "_").upper() or "RELATED_TO"
                try:
                    s.run(
                        f"CREATE INDEX rel_{clean.lower()}_edge_id IF NOT EXISTS "
                        f"FOR ()-[r:{clean}]-() ON (r.edge_id)"
                    )
                except Exception as e:
                    logger.warning(f"Rel index for {clean} skipped: {e}")
        logger.info(f"[{domain}] Relationship edge_id indexes created")

    def _create_qdrant_payload_indexes(self):
        q = self._q()
        indexes = {
            "arxiv_nodes":  ["domain", "entity_type"],
            "arxiv_chunks": ["domain", "rel_type", "src_id", "dst_id"],
            "arxiv_papers": ["domain", "arxiv_id"],
        }
        for coll, fields in indexes.items():
            for f in fields:
                try:
                    q.create_payload_index(
                        collection_name=coll,
                        field_name=f,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.warning(f"Payload index {coll}.{f} skipped: {e}")
        logger.info("Qdrant payload indexes created")

    # ------------------------------------------------------------------
    # Phase 1 — nodes + paper nodes
    # ------------------------------------------------------------------

    def _ingest_nodes(self, domain: str) -> int:
        from collections import defaultdict
        total = 0
        batch: list[NodeRecord] = []

        def _flush():
            nonlocal total
            # Group by entity_type — Cypher requires label literals, so one query per type
            by_type: dict[str, list[dict]] = defaultdict(list)
            for r in batch:
                by_type[r.entity_type].append({
                    "node_id": r.node_id,
                    "domain": r.domain,
                    "display_name": r.display_name,
                    "entity_type": r.entity_type,
                })
            with self._neo4j().session() as s:
                for entity_type, records in by_type.items():
                    s.run(
                        f"UNWIND $batch AS n "
                        f"CREATE (:{entity_type}:_Embeddable "
                        f"{{node_id: n.node_id, domain: n.domain, "
                        f"display_name: n.display_name, entity_type: n.entity_type}})",
                        {"batch": records},
                    )
            embeddable_recs = [r for r in batch if r.display_name]
            display_names = [r.display_name for r in embeddable_recs]
            if display_names:
                embeddings = embed_texts_with_model(display_names, EMBEDDING_MODEL, batch_size=64)
                qdrant_points = [
                    PointStruct(
                        id=str(r.qdrant_id),
                        vector=emb,
                        payload={
                            "node_id": r.node_id,
                            "domain": r.domain,
                            "display_name": r.display_name,
                            "entity_type": r.entity_type,
                        },
                    )
                    for r, emb in zip(embeddable_recs, embeddings)
                ]
                self._q().upsert(collection_name="arxiv_nodes", points=qdrant_points)
            total += len(batch)
            batch.clear()

        for rec in self.loader.iter_nodes(domain):
            batch.append(rec)
            if len(batch) >= self.neo4j_batch_size:
                _flush()

        if batch:
            _flush()

        return total

    def _ingest_paper_nodes(self, domain: str) -> int:
        total = 0
        batch = []

        def _flush():
            nonlocal total
            with self._neo4j().session() as s:
                s.run(
                    "UNWIND $batch AS p "
                    "CREATE (:Paper {arxiv_id: p.arxiv_id, title: p.title, domain: p.domain, "
                    "categories: p.categories, published: p.published})",
                    {"batch": batch},
                )
            total += len(batch)
            batch.clear()

        for rec in self.loader.iter_papers(domain):
            batch.append({
                "arxiv_id": rec.arxiv_id,
                "title": rec.title,
                "domain": rec.domain,
                "categories": rec.categories,
                "published": rec.published,
            })
            if len(batch) >= self.neo4j_batch_size:
                _flush()

        if batch:
            _flush()

        return total

    # ------------------------------------------------------------------
    # Phase 2 — edges
    # ------------------------------------------------------------------

    def _ingest_edges(self, domain: str) -> int:
        total = 0
        # Group by rel_type for batching
        from collections import defaultdict
        buckets: dict[str, list] = defaultdict(list)

        for rec in self.loader.iter_edges(domain):
            buckets[rec.rel_type].append({
                "edge_id": rec.edge_id,
                "src_id": rec.src_id,
                "dst_id": rec.dst_id,
                "domain": rec.domain,
            })

        for rel_type, records in buckets.items():
            for i in range(0, len(records), self.neo4j_batch_size):
                b = records[i : i + self.neo4j_batch_size]
                with self._neo4j().session() as s:
                    s.run(
                        f"UNWIND $batch AS rel "
                        f"MATCH (src:_Embeddable {{node_id: rel.src_id, domain: rel.domain}}) "
                        f"MATCH (dst:_Embeddable {{node_id: rel.dst_id, domain: rel.domain}}) "
                        f"CREATE (src)-[:{rel_type} {{edge_id: rel.edge_id, domain: rel.domain}}]->(dst)",
                        {"batch": b},
                    )
                total += len(b)

        return total

    # ------------------------------------------------------------------
    # Phase 3 — chunks
    # ------------------------------------------------------------------

    def _ingest_chunks(self, domain: str, include_src_names: bool, progress: _Progress) -> int:
        total = 0
        batch_index = 0
        points: list[PointStruct] = []

        def _flush():
            nonlocal total, batch_index
            # Snapshot IDs and Neo4j params before any mutation of points list
            snapshot_ids = [str(p.id) for p in points]
            neo4j_batch = [
                {
                    "src_id": int(p.payload.get("src_id", 0)),
                    "dst_id": int(p.payload.get("dst_id", 0)),
                    "edge_id": int(p.payload.get("edge_id", 0)),
                    "domain": p.payload.get("domain", ""),
                    "qdrant_id": str(p.id),  # UUID string — must be str()
                }
                for p in points
                if p.payload.get("src_id") is not None and p.payload.get("dst_id") is not None
            ]

            qdrant_ok = False
            try:
                self._q().upsert(collection_name="arxiv_chunks", points=points)
                total += len(points)
                qdrant_ok = True
            except Exception as e:
                progress.log_batch_failure(domain, "chunks", batch_index, str(e), snapshot_ids)
                logger.error(f"[{domain}] chunks batch {batch_index} Qdrant upsert failed: {e}")

            if qdrant_ok:
                # Write HAS_CHUNK edges — separate try/except so Qdrant success is not lost
                try:
                    with self._neo4j().session() as s:
                        s.run(
                            """
                            UNWIND $batch AS c
                            MATCH (src:_Embeddable {node_id: c.src_id, domain: c.domain})
                            MATCH (dst:_Embeddable {node_id: c.dst_id, domain: c.domain})
                            MERGE (src)-[r:HAS_CHUNK {edge_id: c.edge_id, domain: c.domain}]->(dst)
                            SET r.qdrant_id = c.qdrant_id
                            """,
                            {"batch": neo4j_batch},
                        )
                except Exception as e:
                    progress.log_batch_failure(domain, "chunks_has_chunk", batch_index, str(e), snapshot_ids)
                    logger.error(f"[{domain}] chunks batch {batch_index} HAS_CHUNK write failed: {e}")

            # Always advance batch_index and clear — exactly once per _flush() call
            batch_index += 1
            points.clear()

        for rec in self.loader.iter_chunks(domain, include_src_names=include_src_names):
            payload = {
                "edge_id": rec.edge_id,
                "src_id": rec.src_id,
                "dst_id": rec.dst_id,
                "rel_type": rec.rel_type,
                "paragraph": rec.paragraph,
                "domain": rec.domain,
                "paper_id": rec.paper_id,
            }
            if rec.src_name is not None:
                payload["src_name"] = rec.src_name
            if rec.dst_name is not None:
                payload["dst_name"] = rec.dst_name

            points.append(PointStruct(
                id=str(rec.qdrant_id),
                vector=rec.embedding,
                payload=payload,
            ))
            if len(points) >= QDRANT_BATCH:
                _flush()

        if points:
            _flush()

        return total

    # ------------------------------------------------------------------
    # Phase 4 — paper chunks (with SPECTER2 embedding)
    # ------------------------------------------------------------------

    def _ingest_paper_chunks(self, domain: str, progress: _Progress) -> int:
        total = 0
        batch_index = 0
        recs: list[PaperChunkRecord] = []

        def _flush():
            nonlocal total, batch_index
            texts = [r.chunk_text for r in recs]
            try:
                embeddings = embed_texts_with_model(texts, EMBEDDING_MODEL, batch_size=64)
                points = [
                    PointStruct(
                        id=str(r.qdrant_id),
                        vector=emb,
                        payload={
                            "arxiv_id": r.arxiv_id,
                            "chunk_index": r.chunk_index,
                            "chunk_text": r.chunk_text,
                            "domain": r.domain,
                            "title": r.title,
                        },
                    )
                    for r, emb in zip(recs, embeddings)
                ]
                self._q().upsert(collection_name="arxiv_papers", points=points)
                total += len(recs)
            except Exception as e:
                ids = [r.arxiv_id for r in recs]
                progress.log_batch_failure(domain, "papers", batch_index, str(e), ids)
                logger.error(f"[{domain}] papers batch {batch_index} failed: {e}")
            batch_index += 1
            recs.clear()

        for rec in self.loader.iter_paper_chunks(domain):
            recs.append(rec)
            if len(recs) >= QDRANT_BATCH:
                _flush()

        if recs:
            _flush()

        return total

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        reset: bool = False,
        domains: Optional[list[str]] = None,
        include_src_names: bool = False,
    ) -> dict:
        """
        Run the full ingestion pipeline.

        Args:
            reset: Wipe Neo4j + Qdrant + progress file before starting
            domains: Subset of domains to process (default: all three)
            include_src_names: Populate src_name/dst_name in arxiv_chunks payload

        Returns:
            Summary dict
        """
        domains = domains or DOMAINS

        # --- Startup ---
        logger.info("Validating parquet files...")
        self.loader.validate()

        logger.info("Checking connectivity...")
        self._ping_neo4j()
        self._ping_qdrant()
        self._check_sentence_transformers()

        if reset:
            self._reset()

        self._ensure_collections()
        self._create_neo4j_indexes()
        self._create_qdrant_payload_indexes()

        progress = _Progress(self.progress_file)

        summary = {}

        for domain in domains:
            logger.info(f"=== Domain: {domain} ===")
            domain_stats = {}

            # Phase 1: nodes
            if progress.is_done(domain, "nodes"):
                logger.info(f"[{domain}] nodes phase already complete, skipping")
            else:
                try:
                    n = self._ingest_nodes(domain)
                    p = self._ingest_paper_nodes(domain)
                    domain_stats["nodes"] = n
                    domain_stats["paper_nodes"] = p
                    progress.mark_done(domain, "nodes")
                    logger.info(f"[{domain}] nodes phase done: {n} entity nodes, {p} paper nodes")
                except Exception as e:
                    progress.mark_failed(domain, "nodes")
                    logger.error(f"[{domain}] nodes phase FAILED: {e}")
                    summary[domain] = {"error": str(e), "phase": "nodes"}
                    continue

            # Phase 2: edges
            if progress.is_done(domain, "edges"):
                logger.info(f"[{domain}] edges phase already complete, skipping")
            else:
                try:
                    n = self._ingest_edges(domain)
                    domain_stats["edges"] = n
                    progress.mark_done(domain, "edges")
                    self._create_rel_indexes(domain)
                    logger.info(f"[{domain}] edges phase done: {n} edges")
                except Exception as e:
                    progress.mark_failed(domain, "edges")
                    logger.error(f"[{domain}] edges phase FAILED: {e}")
                    summary[domain] = {"error": str(e), "phase": "edges"}
                    continue

            # Phase 3: chunks
            if progress.is_done(domain, "chunks"):
                logger.info(f"[{domain}] chunks phase already complete, skipping")
            else:
                n = self._ingest_chunks(domain, include_src_names, progress)
                domain_stats["chunks"] = n
                progress.mark_done(domain, "chunks")
                logger.info(f"[{domain}] chunks phase done: {n} chunk points")

            # Phase 4: paper chunks
            if progress.is_done(domain, "papers"):
                logger.info(f"[{domain}] papers phase already complete, skipping")
            else:
                n = self._ingest_paper_chunks(domain, progress)
                domain_stats["paper_chunks"] = n
                progress.mark_done(domain, "papers")
                logger.info(f"[{domain}] papers phase done: {n} paper chunk points")

            summary[domain] = domain_stats

        self.close()
        logger.info(f"Ingestion complete: {summary}")
        return summary
