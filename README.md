# Hybrid RAG — AI, Computers & Society, Quantitative Methods arXiv Papers

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

## Evaluation Results

Evaluated across 30 questions per mode (6 question types × 5 questions, 3 arXiv domains). Metrics computed with an LLM judge for answer correctness and RAGAS-style context recall.

### Overall (n=30)

| Metric | Vector-only | Hybrid | Delta |
|---|---|---|---|
| Context Recall | 0.307 | **0.388** | +26% |
| Faithfulness | 0.825 | 0.717 | — |
| Answer Correctness | 0.393 | 0.386 | ~= |
| Refusal Rate | 56.7% | **33.3%** | -41% |

Hybrid mode retrieves more relevant context (+26% recall) and refuses significantly fewer questions (-41%), at a modest faithfulness tradeoff.

### Answer Correctness by Question Type

| Question Type | Vector | Hybrid |
|---|---|---|
| Counterfactual | 0.52 | **0.68** |
| Open-ended | 0.40 | **0.42** |
| Single-hop | **0.48** | 0.43 |
| Single-hop w/ conditions | 0.44 | 0.44 |
| Multi-hop | **0.31** | 0.27 |
| Multi-hop difficult | **0.20** | 0.07 |

### Context Recall by Question Type

| Question Type | Vector | Hybrid |
|---|---|---|
| Open-ended | 0.39 | **0.62** |
| Multi-hop | 0.50 | **0.58** |
| Counterfactual | 0.38 | **0.47** |
| Multi-hop difficult | 0.00 | **0.09** |
| Single-hop w/ conditions | 0.49 | 0.49 |
| Single-hop | 0.08 | 0.08 |

Hybrid retrieval gains are strongest on open-ended (+59% recall) and counterfactual (+31% correctness) questions, where graph entity context supplements dense vector results. Multi-hop difficult questions remain a known hard case for both modes.

## License

MIT
