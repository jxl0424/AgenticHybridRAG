"""
FastAPI server with Inngest event-driven RAG functions.
"""
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid 
import os
import datetime

from pydantic import BaseModel
from src.ingestion.pdf_loader import load_and_chunk_pdf, embed_texts
from src.retrieval.qdrant_storage import QdrantStorage
from src.graph.knowledge_graph import get_knowledge_graph, GraphDocument, GraphChunk
from src.graph.entity_extractor import get_entity_extractor
from src.pipelines.graph_rag_pipeline import GraphRAGPipeline
from src.generation.llm_client import LLMClient
from src.types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from src.prompts.templates import build_messages

load_dotenv()

# Initialize our core components once for the worker
_llm_client = LLMClient()
_kg = get_knowledge_graph()
_extractor = get_entity_extractor(llm_client=_llm_client)
_qdrant = QdrantStorage()
_pipeline = GraphRAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="GraphRAG: Extract & Ingest Chunk",
    trigger=inngest.TriggerEvent(event="rag/process_chunk"),
    retries=5
)
async def process_chunk_workflow(ctx: inngest.Context):
    """
    Background worker that extracts entities from a single chunk using the LLM.
    If the GitHub Models API hits a rate limit, Inngest will automatically 
    catch the failure, back off, and retry this specific chunk.
    """
    chunk_text = ctx.event.data["text"]
    chunk_id = ctx.event.data["chunk_id"]
    doc_id = ctx.event.data["doc_id"]
    chunk_index = ctx.event.data["chunk_index"]

    def _extract_and_save():
        # First, extract entities and relationships
        extraction = _extractor.extract(chunk_text)
        
        # Second, save to Knowledge Graph
        graph_chunk = GraphChunk(
            id=chunk_id,
            text=chunk_text,
            document_id=doc_id,
            chunk_index=chunk_index
        )
        _kg.ingest_extraction_result(graph_chunk, extraction)
        
        # Third, save Vector Embedding to Qdrant
        vec = embed_texts([chunk_text])[0]
        _qdrant.upsert(
            [chunk_id], 
            [vec], 
            [{"source": doc_id, "text": chunk_text}]
        )
        
        return {
            "entities_found": len(extraction.entities),
            "relationships_found": len(extraction.relationships)
        }

    # Wrap the entire synchronous extraction in a step
    result = await ctx.step.run("extract-and-save", _extract_and_save)
    return result


@inngest_client.create_function(
    fn_id="GraphRAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def process_document_workflow(ctx: inngest.Context):
    """
    Main orchestrator workflow. When a PDF is uploaded, this loads the text,
    creates the Document node, and fans out 'process_chunk' events.
    """
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", os.path.basename(pdf_path))

    def _load_and_init_doc():
        chunks = load_and_chunk_pdf(pdf_path)
        doc_id = source_id
        
        _kg.add_document(GraphDocument(
            id=doc_id,
            title=source_id,
            source=pdf_path
        ))
        
        return chunks, doc_id

    # 1. Load document and chunk it
    chunks, doc_id = await ctx.step.run("load-and-chunk", _load_and_init_doc)

    # 2. Fan out to process each chunk dynamically in parallel
    events = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}"))
        events.append(
            inngest.Event(
                name="rag/process_chunk",
                data={
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": i
                }
            )
        )
        
    # Send all events to Inngest to be processed by our chunk worker (with retries!)
    if events:
        await ctx.step.send_event("fan-out-chunks", events)

    return {"document": doc_id, "chunks_queued": len(events)}


@inngest_client.create_function(
    fn_id="GraphRAG: Query Pipeline",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def query_pipeline_workflow(ctx: inngest.Context):
    """
    Handles queries using the full GraphRAG Hybrid Pipeline.
    """
    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)
    use_hybrid = ctx.event.data.get("use_hybrid", True)
    
    def _run_pipeline():
        pipeline = GraphRAGPipeline()
        return pipeline.query_with_sources(question, top_k=top_k, use_hybrid=use_hybrid)

    result = await ctx.step.run(f"{'hybrid' if use_hybrid else 'vector'}-query", _run_pipeline)
    
    return {
        "answer": result["answer"], 
        "sources": result["sources"], 
        "vector_contexts": result["vector_contexts"],
        "graph_contexts": result["graph_contexts"],
        "entities_found": result["entities_found"],
        "retrieval_type": result["retrieval_type"],
        "thought_process": result.get("thought_process", [])
    }

app = FastAPI()

@app.post("/api/query")
def query_endpoint(req: QueryRequest):
    try:
        return _pipeline.query_with_sources(
            req.question, top_k=req.top_k, use_hybrid=req.use_hybrid
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

inngest.fast_api.serve(
    app,
    inngest_client,
    [process_document_workflow, process_chunk_workflow, query_pipeline_workflow],
    serve_origin="http://host.docker.internal:8000"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)


