# Medical Graph RAG

A **Hybrid Retrieval-Augmented Generation** system for medical PDF documents, combining:

- **Semantic search** via Qdrant (vector DB)
- **Knowledge graph** via Neo4j (entity relationships: diseases, drugs, symptoms...)
- **Medical embeddings** via `pritamdeka/S-PubMedBert-MS-MARCO` (PubMed-trained)
- **PDF ingestion** with medical NER entity extraction

## Quick Start

### 1. Start Infrastructure (Docker)

```powershell
docker compose up -d
```

This starts:
| Service | URL | Notes |
|---------|-----|-------|
| Neo4j Browser | http://localhost:7474 | Graph visualization UI |
| Neo4j Bolt | bolt://localhost:7687 | Login: `neo4j` / `your_password` |
| Qdrant REST | http://localhost:6333 | Vector database API |

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Ingest PDFs

Place PDFs in `data/raw/`, then run:

```powershell
# Fresh ingest (first time or to rebuild)
python ingest_graph_rag.py --recreate

# Incremental (add new PDFs only)
python ingest_graph_rag.py
```

### 4. Start the API Server

```powershell
python -m src.api.server
```

### 5. Start the Streamlit UI

```powershell
streamlit run src/api/app.py
```

## Architecture

```
data/raw/*.pdf
     │
     ▼
[PDF Loader + Chunker]
     │
     ├──► [Medical Embedding]    ───► Qdrant (vector store)
     │    S-PubMedBert-MS-MARCO
     │
     └──► [Medical NER]          ───► Neo4j (knowledge graph)
          Rule-based extractor        DISEASE, DRUG, SYMPTOM...
                                      TREATS, CAUSES, HAS_PROCEDURE
                                           │
                 ┌─────────────────────────┘
                 ▼
         [Hybrid Retriever]
         Vector (60%) + Graph (40%)
                 │
                 ▼
           [LLM Answer]
           Local LLM (Llama3.2 8b) or
           OpenAI Models (gpt-4o-mini)
```

## Project Structure

```
RAG/
├── docker-compose.yml      # Neo4j + Qdrant services
├── config/defaults.yaml    # All configuration
├── data/raw/               # Drop PDFs here
├── ingest_graph_rag.py     # Main ingestion script
└── src/
    ├── ingestion/          # PDF loader + medical embeddings
    ├── graph/              # Entity extractor + knowledge graph
    ├── retrieval/          # Qdrant, graph, hybrid retrieval
    ├── pipelines/          # GraphRAGPipeline (end-to-end)
    ├── generation/         # LLM client
    └── api/                # FastAPI server + Streamlit UI
```

## Neo4j Graph Schema

After ingestion, query the graph at http://localhost:7474:

```cypher
// See all medical entities
MATCH (e:MedicalEntity) RETURN e LIMIT 25

// Drug → Disease relationships
MATCH (d:DRUG)-[:TREATS]->(dis:DISEASE) RETURN d, dis LIMIT 10

// What chunks mention hypertension?
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:MedicalEntity {name: "hypertension"})
RETURN c.text LIMIT 5
```

## Environment Variables

Copy `.env-sample` → `.env` and set:

| Variable | Description |
|----------|-------------|
| `API_KEY` | API key for LLM |
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | `neo4j` |
| `NEO4J_PASSWORD` | Your Neo4j password |
| `QDRANT_URL` | `http://localhost:6333` |

## License

MIT
