"""
Diagnostic script for LocalIngestionPipeline — checks each layer independently.

Run from project root:
    python diagnose_hybridrag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEP = "-" * 60

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Pokemon0424$")
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "allenai/specter2_base"
COLLECTIONS = ["arxiv_nodes", "arxiv_chunks", "arxiv_papers"]


def check_qdrant_collections():
    print(SEP)
    print("CHECK 1 — Qdrant collections and point counts")
    print(SEP)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        existing = {c.name for c in client.get_collections().collections}
        print(f"  Collections found: {sorted(existing)}")
        for name in COLLECTIONS:
            if name in existing:
                count = client.count(name).count
                print(f"  {name}: {count} points")
                if count == 0:
                    print(f"  WARNING: '{name}' is EMPTY")
            else:
                print(f"  WARNING: '{name}' does NOT exist")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def check_qdrant_sample():
    print(SEP)
    print("CHECK 2 — Sample vectors from 'arxiv_chunks'")
    print(SEP)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        results, _ = client.scroll("arxiv_chunks", limit=3, with_payload=True, with_vectors=True)
        if not results:
            print("  No points found in 'arxiv_chunks'")
            return
        for pt in results:
            vec = pt.vector if pt.vector else []
            payload = pt.payload or {}
            text = payload.get("paragraph", "")[:80]
            print(f"  id={pt.id}  vec_dim={len(vec)}  domain={payload.get('domain')}")
            print(f"    paragraph: {text!r}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def check_qdrant_raw_search():
    print(SEP)
    print("CHECK 3 — Raw vector search on 'arxiv_chunks' (score_threshold=0)")
    print(SEP)
    try:
        from src.ingestion.hf_hybridrag_loader import embed_texts_with_model
        from qdrant_client import QdrantClient

        print(f"  Embedding query with {EMBEDDING_MODEL} ...")
        vec = embed_texts_with_model(["what is the attention mechanism"], EMBEDDING_MODEL)[0]
        print(f"  Query vector dim: {len(vec)}")

        client = QdrantClient(url=QDRANT_URL, timeout=10)
        results = client.query_points(
            collection_name="arxiv_chunks",
            query=vec,
            limit=5,
            score_threshold=0.0,
        )
        points = results.points
        if not points:
            print("  No results — collection empty or dimension mismatch")
        else:
            for r in points:
                payload = r.payload or {}
                text = payload.get("paragraph", "")[:80]
                print(f"  score={r.score:.4f}  domain={payload.get('domain')}  paragraph={text!r}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def check_neo4j():
    print(SEP)
    print("CHECK 4 — Neo4j node and relationship counts")
    print(SEP)
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            embeddable = session.run("MATCH (n:_Embeddable) RETURN count(n) AS n").single()["n"]
            paper = session.run("MATCH (p:Paper) RETURN count(p) AS n").single()["n"]
            rel = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
            print(f"  _Embeddable nodes : {embeddable}")
            print(f"  Paper nodes       : {paper}")
            print(f"  Relationships     : {rel}")
            if embeddable == 0 and paper == 0:
                print("  WARNING: Neo4j graph is EMPTY")
            if embeddable > 0:
                rows = session.run(
                    "MATCH (n:_Embeddable) RETURN n.entity_type AS t, count(n) AS c "
                    "ORDER BY c DESC LIMIT 10"
                ).data()
                print("  Entity type breakdown:")
                for row in rows:
                    print(f"    {row['t']}: {row['c']}")
        driver.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def check_neo4j_relationships():
    print(SEP)
    print("CHECK 5 — Neo4j relationship types")
    print(SEP)
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            rows = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC LIMIT 10"
            ).data()
            if not rows:
                print("  No relationships found")
            else:
                for row in rows:
                    print(f"  {row['t']}: {row['c']}")
        driver.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def check_end_to_end_query():
    print(SEP)
    print("CHECK 6 — End-to-end: embed query, search arxiv_chunks + arxiv_papers")
    print(SEP)
    try:
        from src.ingestion.hf_hybridrag_loader import embed_texts_with_model
        from qdrant_client import QdrantClient

        question = "what is the attention mechanism"
        print(f"  Query: {question!r}")

        vec = embed_texts_with_model([question], EMBEDDING_MODEL, batch_size=1)[0]
        print(f"  Vector dim: {len(vec)}")

        client = QdrantClient(url=QDRANT_URL, timeout=10)

        chunk_results = client.query_points(
            collection_name="arxiv_chunks",
            query=vec,
            limit=3,
            score_threshold=0.0,
        ).points
        print(f"  arxiv_chunks hits: {len(chunk_results)}")
        for r in chunk_results:
            text = (r.payload or {}).get("paragraph", "")[:100]
            print(f"    score={r.score:.4f}  {text!r}")

        paper_results = client.query_points(
            collection_name="arxiv_papers",
            query=vec,
            limit=3,
            score_threshold=0.0,
        ).points
        print(f"  arxiv_papers hits: {len(paper_results)}")
        for r in paper_results:
            text = (r.payload or {}).get("chunk_text", "")[:100]
            print(f"    score={r.score:.4f}  {text!r}")

    except Exception as e:
        print(f"  ERROR: {e}")
    print()


if __name__ == "__main__":
    print("HybridRAG Diagnostic (LocalIngestionPipeline)")
    print("=" * 60)
    print()
    check_qdrant_collections()
    check_qdrant_sample()
    check_qdrant_raw_search()
    check_neo4j()
    check_neo4j_relationships()
    check_end_to_end_query()
    print("=" * 60)
    print("Diagnostic complete.")
