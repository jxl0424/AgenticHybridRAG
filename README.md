# Hybrid RAG — CS/AI arXiv Papers

A **Hybrid Retrieval-Augmented Generation** system for CS/AI research papers from arXiv, combining:

- **Semantic vector search** via Qdrant (paragraph-level and paper-level)
- **Knowledge graph retrieval** via Neo4j (CS/AI entities: models, datasets, tasks, algorithms, venues)
- **SPECTER2 embeddings** (`allenai/specter2_base`, 768-dim)
- **HybridRAG-Bench** dataset (arxiv_ai, arxiv_cy, arxiv_qm domains)
- **LLM generation** via OpenRouter (free models) with Ollama fallback

## Quick Start

### 1. Start Infrastructure

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| Neo4j Browser | http://localhost:7474 |
| Neo4j Bolt | bolt://localhost:7687 |
| Qdrant REST | http://localhost:6333 |
| Inngest Dev | http://localhost:8288 |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env-sample` → `.env` and set:

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Enables OpenRouter LLM provider |
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | `neo4j` |
| `NEO4J_PASSWORD` | Your Neo4j password |
| `QDRANT_URL` | `http://localhost:6333` |

### 4. Ingest Data

```bash
# Download HybridRAG-Bench parquet files into data/hybridrag/ first, then:
python ingest_local.py                          # Resume from last checkpoint
python ingest_local.py --reset                  # Wipe all data and re-ingest
python ingest_local.py --domains arxiv_ai arxiv_cy  # Specific domains only
```

### 5. Query

```bash
python query_hybridrag.py "What is the attention mechanism in transformers?"
python query_hybridrag.py "question" --compare  # Hybrid vs vector-only comparison
```

### 6. Streamlit UI

```bash
streamlit run src/api/app.py
```

## Architecture

```
Query
  │
  ▼
SPECTER2 embed
  │
  ├──► ChunkRetriever   (Qdrant: arxiv_chunks — paragraph-level)
  ├──► PaperRetriever   (Qdrant: arxiv_papers — full-paper)
  └──► GraphRetriever   (CSEntityExtractor → Neo4j → Qdrant chunk fetch)
         │
         ▼
    HybridRetriever  (RRF fusion: chunk 0.2 · paper 0.6 · graph 0.2)
         │
         ▼
    Reranker         (cross-encoder msmarco + relative threshold)
         │
         ▼
    LLMClient        (OpenRouter → Ollama fallback)
         │
         ▼
      Answer
```

## Qdrant Collections

| Collection | Content | Dim |
|---|---|---|
| `arxiv_chunks` | Paragraph embeddings | 768 |
| `arxiv_papers` | Full-paper embeddings | 768 |
| `arxiv_nodes` | CSEntity embeddings (for semantic entity lookup) | 768 |

## Neo4j Graph Schema

```cypher
// CS entities (models, datasets, tasks, algorithms, venues, ...)
MATCH (e:CSEntity) RETURN e LIMIT 25

// Chunks linked to an entity
MATCH (e:CSEntity {name: "BERT"})-[:HAS_CHUNK]->(c) RETURN c LIMIT 10

// Relationships between entities
MATCH (m:CSEntity)-[r]->(d:CSEntity) WHERE m.entity_type = "MODEL" RETURN m, r, d LIMIT 20
```

## Project Structure

```
RAG/
├── docker-compose.yml
├── config/defaults.yaml          # All tunable parameters
├── data/hybridrag/               # HybridRAG-Bench parquet files
├── ingest_local.py               # Ingestion entry point
├── query_hybridrag.py            # Query entry point
├── diagnose_hybridrag.py         # Health checks + sample queries
└── src/
    ├── graph/                    # CSEntityExtractor + CSKnowledgeGraph
    ├── ingestion/                # LocalParquetLoader
    ├── retrieval/                # ChunkRetriever, PaperRetriever, GraphRetriever, HybridRetriever
    ├── pipelines/                # HybridRAGBenchPipeline, LocalIngestionPipeline
    ├── generation/               # LLMClient (OpenRouter + Ollama)
    ├── prompts/                  # System prompt + RAG query template
    ├── observability/            # Arize Phoenix tracing (optional)
    └── api/                     # Streamlit UI
```

## Tests

```bash
pytest tests/unit/ -v                          # Unit tests (no services needed)
pytest tests/integration/ -v                   # Integration tests (requires Docker)
pytest tests/evaluation/hybridrag_eval.py -v   # End-to-end eval with RAGAS metrics
```

## License

MIT
