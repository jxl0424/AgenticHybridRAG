"""
One-shot migration: create HAS_CHUNK edges in Neo4j for already-ingested data.

Reads parquet files from data/hybridrag/ (same source as LocalIngestionPipeline),
recomputes qdrant_id = uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}"), and writes
MERGE-idempotent HAS_CHUNK edges between _Embeddable nodes.

Safe to re-run — uses MERGE + unconditional SET so partial runs are healed.

Usage:
    python scripts/migrate_has_chunk_edges.py
    python scripts/migrate_has_chunk_edges.py --data-dir data/hybridrag --batch-size 500
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.local_parquet_loader import LocalParquetLoader, DOMAINS, HYBRIDRAG_NS
from src.utils import get_logger

logger = get_logger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pokemon0424$"
BATCH_SIZE = 500

CYPHER = """
UNWIND $batch AS c
MATCH (src:_Embeddable {node_id: c.src_id, domain: c.domain})
MATCH (dst:_Embeddable {node_id: c.dst_id, domain: c.domain})
MERGE (src)-[r:HAS_CHUNK {edge_id: c.edge_id, domain: c.domain}]->(dst)
SET r.qdrant_id = c.qdrant_id
"""


def migrate(data_dir: str, batch_size: int, domains: list[str]):
    from neo4j import GraphDatabase

    loader = LocalParquetLoader(data_dir)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    total = 0
    errors = 0

    try:
        for domain in domains:
            logger.info(f"[{domain}] Starting HAS_CHUNK migration...")
            batch = []

            for rec in loader.iter_chunks(domain):
                # Use rec.qdrant_id directly — ChunkRecord.__post_init__ computes it via
                # uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}"), same formula as LocalIngestionPipeline
                batch.append({
                    "src_id": rec.src_id,
                    "dst_id": rec.dst_id,
                    "edge_id": rec.edge_id,
                    "domain": domain,
                    "qdrant_id": str(rec.qdrant_id),  # always str — rec.qdrant_id is uuid.UUID
                })

                if len(batch) >= batch_size:
                    try:
                        with driver.session() as s:
                            s.run(CYPHER, {"batch": batch})
                        total += len(batch)
                        logger.info(f"[{domain}] {total} edges written...")
                    except Exception as e:
                        errors += len(batch)
                        logger.error(f"[{domain}] Batch failed: {e}")
                    batch = []

            if batch:
                try:
                    with driver.session() as s:
                        s.run(CYPHER, {"batch": batch})
                    total += len(batch)
                except Exception as e:
                    errors += len(batch)
                    logger.error(f"[{domain}] Final batch failed: {e}")

            logger.info(f"[{domain}] Done.")

    finally:
        driver.close()

    print(f"\nMigration complete: {total} edges written, {errors} errors.")
    if errors:
        print("Re-run to retry failed batches — MERGE is idempotent.")


def main():
    parser = argparse.ArgumentParser(description="Migrate HAS_CHUNK edges into Neo4j")
    parser.add_argument("--data-dir", default="data/hybridrag")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--domains", nargs="+", default=DOMAINS,
        choices=DOMAINS,
    )
    args = parser.parse_args()
    migrate(args.data_dir, args.batch_size, args.domains)


if __name__ == "__main__":
    main()
