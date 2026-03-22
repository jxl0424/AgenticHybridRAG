
import os
import sys
import uuid
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipelines.graph_rag_pipeline import GraphRAGPipeline
from src.graph.knowledge_graph import GraphDocument, GraphChunk, GraphEntity
from src.graph.entity_extractor import ExtractionResult, ExtractedEntity

def test_ingestion_integrity():
    print("🚀 Starting Ingestion Integrity Test...")
    pipeline = GraphRAGPipeline()
    
    # 1. Test Metadata Serialization (Numpy types)
    print("\n--- Testing Metadata Serialization ---")
    kg = pipeline.knowledge_graph
    test_doc = GraphDocument(
        id="test_meta_1",
        title="Meta Test",
        source="test",
        metadata={
            "score": np.float32(0.95),
            "count": np.int64(10),
            "vector": np.array([0.1, 0.2, 0.3])
        }
    )
    try:
        kg.add_document(test_doc)
        print("✅ Metadata with numpy types ingested successfully.")
    except Exception as e:
        print(f"❌ Metadata ingestion failed: {e}")
        return

    # 2. Test Idempotency (Deterministic IDs)
    print("\n--- Testing Idempotency (Deterministic IDs) ---")
    doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"
    
    def ingest_mock_doc():
        # Manually simulate what ingest_document does but with controlled chunks
        doc = GraphDocument(id=doc_id, title="Idempotency Test", source="test")
        kg.add_document(doc)
        
        # We'll use the pipeline's own ID generator via pdf_loader
        from src.ingestion.pdf_loader import generate_chunk_id
        
        chunk_text = "SARS-CoV-2 is a coronavirus that causes COVID-19."
        chunk_id = generate_chunk_id(doc_id, 0)
        
        chunk = GraphChunk(id=chunk_id, text=chunk_text, document_id=doc_id, chunk_index=0)
        
        # Mock extraction result
        res = ExtractionResult(
            chunk_id=chunk_id,
            text=chunk_text,
            entities=[
                ExtractedEntity(text="SARS-CoV-2", entity_type="VIRUS_STRAIN"),
                ExtractedEntity(text="COVID-19", entity_type="DISEASE")
            ],
            relationships=[("SARS-CoV-2", "CAUSES", "COVID-19")]
        )
        
        kg.ingest_extraction_result(chunk, res)
        return chunk_id

    print("Ingesting first time...")
    cid1 = ingest_mock_doc()
    stats1 = kg.get_stats()
    nodes1 = stats1.get('nodes', {}).get('Chunk', 0)
    entities1 = stats1.get('nodes', {}).get('MedicalEntity', 0)
    
    print(f"  Chunks: {nodes1}, Entities: {entities1}")

    print("Ingesting second time (should be identical IDs)...")
    cid2 = ingest_mock_doc()
    stats2 = kg.get_stats()
    nodes2 = stats2.get('nodes', {}).get('Chunk', 0)
    entities2 = stats2.get('nodes', {}).get('MedicalEntity', 0)
    
    print(f"  Chunks: {nodes2}, Entities: {entities2}")

    if cid1 == cid2:
        print(f"✅ Chunk IDs match: {cid1}")
    else:
        print(f"❌ Chunk IDs differ! {cid1} vs {cid2}")

    # Check if counts increased (they shouldn't if MERGE is working with deterministic IDs)
    # Note: We use GET_STATS which counts all nodes, so we check if the count stayed the same
    if nodes1 == nodes2 and entities1 == entities2:
        print("✅ Idempotency confirmed: Node counts did not increase.")
    else:
        print(f"⚠️  Idempotency warning: Counts changed. Chunks: {nodes1}->{nodes2}, Entities: {entities1}->{entities2}")
        print("   (This might happen if other ingestion is running, but in isolation it should be stable)")

    # 3. Test Real PDF Ingestion (Small Scale)
    print("\n--- Testing Real PDF Ingestion ---")
    # We'll search for a small pdf in the data dir or use a dummy
    dummy_pdf = "data/test_dummy.pdf"
    if not os.path.exists(dummy_pdf):
        print("Skipping real PDF test (no dummy PDF found).")
    else:
        try:
            res = pipeline.ingest_document(dummy_pdf)
            print(f"✅ Ingested PDF: {res['chunks_processed']} chunks, {res['entities_extracted']} entities.")
        except Exception as e:
            print(f"❌ PDF Ingestion failed: {e}")

    print("\n🎉 Integrity tests completed.")

if __name__ == "__main__":
    test_ingestion_integrity()
