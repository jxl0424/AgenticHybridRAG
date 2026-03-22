"""
CLI entry point for local HybridRAG-Bench ingestion.

Usage:
    python ingest_local.py [--reset] [--domains arxiv_ai arxiv_cy arxiv_qm]
                           [--src-names] [--batch-size 500]

WARNING: --reset wipes ALL nodes and relationships from Neo4j (including
data from other pipelines) and drops/recreates all three Qdrant collections.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipelines.local_ingestion_pipeline import LocalIngestionPipeline
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest HybridRAG-Bench from local parquet files into Neo4j + Qdrant"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Wipe Neo4j (MATCH (n) DETACH DELETE n), drop and recreate all 3 Qdrant collections, "
            "and delete the progress file. WARNING: removes ALL data from Neo4j."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["arxiv_ai", "arxiv_cy", "arxiv_qm"],
        choices=["arxiv_ai", "arxiv_cy", "arxiv_qm"],
        help="Domains to ingest (default: all three)",
    )
    parser.add_argument(
        "--src-names",
        action="store_true",
        help="Populate src_name/dst_name in arxiv_chunks Qdrant payload",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Neo4j write batch size (default: 500)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/hybridrag",
        help="Path to local HybridRAG-Bench data directory (default: data/hybridrag)",
    )
    args = parser.parse_args()

    if args.reset:
        print(
            "\nWARNING: --reset will wipe ALL Neo4j data and drop all three Qdrant collections.\n"
            "This cannot be undone. Proceed? [y/N] ",
            end="",
            flush=True,
        )
        response = input().strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    pipeline = LocalIngestionPipeline(
        data_dir=args.data_dir,
        neo4j_batch_size=args.batch_size,
    )

    print(f"Starting ingestion: domains={args.domains}, reset={args.reset}, src_names={args.src_names}")
    summary = pipeline.run(
        reset=args.reset,
        domains=args.domains,
        include_src_names=args.src_names,
    )

    print("\nIngestion summary:")
    for domain, stats in summary.items():
        print(f"  {domain}: {stats}")


if __name__ == "__main__":
    main()
