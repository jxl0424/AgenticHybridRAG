"""
Graph RAG Pipeline combining document ingestion, entity extraction, and hybrid retrieval.
"""
import os
import uuid
from typing import Optional, Any
from pathlib import Path

from src.ingestion.pdf_loader import load_and_chunk_pdf, embed_texts
from src.retrieval.qdrant_storage import QdrantStorage
from src.graph.entity_extractor import get_entity_extractor
from src.graph.knowledge_graph import KnowledgeGraph, GraphDocument, GraphChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LLMClient
from src.prompts.templates import build_messages


class GraphRAGPipeline:
    """
    Complete Graph RAG pipeline for document ingestion and querying.
    """
    
    def __init__(
        self,
        config_path: str = "config/defaults.yaml",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    ):
        """
        Initialize the Graph RAG pipeline.
        
        Args:
            config_path: Path to configuration file
            embedding_model: Embedding model to use
        """
        self.embedding_model = embedding_model
        
        # Initialize storage and generation components
        self.qdrant = QdrantStorage()
        self.llm = LLMClient()
        self.entity_extractor = get_entity_extractor(llm_client=self.llm)
        self.knowledge_graph = KnowledgeGraph()
        
        # Wire GraphRetriever to the SAME KnowledgeGraph instance
        # (previously was passed as None which caused a disconnected lazy-init)
        from src.retrieval.graph_retriever import GraphRetriever
        self.graph_retriever = GraphRetriever(self.knowledge_graph)
        
        self.hybrid_retriever = HybridRetriever(
            qdrant_storage=self.qdrant,
            graph_retriever=self.graph_retriever
        )
        
        # Initialize knowledge graph schema
        self.knowledge_graph.initialize_schema()
    
    def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        embed: bool = True
    ) -> dict:
        """
        Ingest a document into both Qdrant and Neo4j.
        
        Args:
            file_path: Path to the PDF document
            document_id: Optional document ID (defaults to filename)
            embed: Whether to generate embeddings for Qdrant
            
        Returns:
            Dictionary with ingestion statistics
        """
        document_id = document_id or os.path.basename(file_path)
        
        # Load and chunk the document
        chunks = load_and_chunk_pdf(file_path)
        
        # Add document to knowledge graph
        doc = GraphDocument(
            id=document_id,
            title=document_id,
            source=file_path,
            metadata={"file_path": file_path}
        )
        self.knowledge_graph.add_document(doc)
        
        # Process each chunk
        total_entities = 0
        total_relationships = 0
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())  # Use UUID for Qdrant compatibility
            
            # Add chunk to knowledge graph
            chunk = GraphChunk(
                id=f"{document_id}_chunk_{i}",  # Keep descriptive ID for KG
                text=chunk_text,
                document_id=document_id,
                chunk_index=i,
                metadata={"source": file_path}
            )
            
            # Extract entities from chunk
            extraction_result = self.entity_extractor.extract(
                chunk_text, 
                chunk_id
            )
            
            # Ingest into knowledge graph
            self.knowledge_graph.ingest_extraction_result(
                chunk, 
                extraction_result
            )
            
            total_entities += len(extraction_result.entities)
            total_relationships += len(extraction_result.relationships)
            
            # Add to Qdrant if embedding
            if embed:
                embeddings = embed_texts([chunk_text])
                
                self.qdrant.upsert(
                    ids=[chunk_id],
                    vectors=[embeddings[0]],
                    payloads=[{
                        "text": chunk_text,
                        "source": file_path,
                        "document_id": document_id
                    }]
                )
        
        return {
            "document_id": document_id,
            "chunks_processed": len(chunks),
            "entities_extracted": total_entities,
            "relationships_extracted": total_relationships
        }
    
    def ingest_documents(
        self,
        directory: str,
        file_pattern: str = "*.pdf"
    ) -> list[dict]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Directory containing documents
            file_pattern: File pattern to match
            
        Returns:
            List of ingestion results
        """
        dir_path = Path(directory)
        results = []
        
        for file_path in dir_path.glob(file_pattern):
            try:
                result = self.ingest_document(str(file_path))
                results.append(result)
                print(f"Ingested: {file_path.name}")
            except Exception as e:
                print(f"Error ingesting {file_path.name}: {e}")
                results.append({
                    "document_id": file_path.name,
                    "error": str(e)
                })
        
        return results
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> dict:
        """
        Query the Graph RAG system.
        
        Args:
            question: Question to answer
            top_k: Number of contexts to retrieve
            use_hybrid: Whether to use hybrid retrieval (vs vector only)
            
        Returns:
            Dictionary with answer and sources
        """
        # Generate query embedding
        query_embedding = embed_texts([question])[0]
        
        if use_hybrid:
            # Use hybrid retrieval
            results = self.hybrid_retriever.retrieve(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            contexts = results.contexts
            sources = results.sources
            entities_found = results.entities_found
        else:
            # Use vector-only retrieval
            results = self.qdrant.search(
                query_vector=query_embedding,
                top_k=top_k
            )
            contexts = results.get("contexts", [])
            sources = results.get("sources", [])
            entities_found = []
        
        # Build prompt and generate answer
        if contexts:
            messages = build_messages(question, contexts)
            answer = self.llm.generate(messages)
        else:
            answer = "I don't have enough information to answer that question based on the provided documents."
        
        return {
            "answer": answer,
            "sources": sources,
            "num_contexts": len(contexts),
            "entities_found": entities_found
        }
    
    def query_with_sources(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> dict:
        """
        Query with detailed source information.
        
        Args:
            question: Question to answer
            top_k: Number of contexts
            use_hybrid: Whether to use hybrid retrieval
            
        Returns:
            Dictionary with detailed results
        """
        # Generate query embedding
        query_embedding = embed_texts([question])[0]
        
        if use_hybrid:
            results = self.hybrid_retriever.retrieve(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            contexts = results.contexts
            sources = results.sources
            scores = results.scores
            vector_contexts = results.vector_contexts
            graph_contexts = results.graph_contexts
            entities_found = results.entities_found
        else:
            results = self.qdrant.search(
                query_vector=query_embedding,
                top_k=top_k
            )
            contexts = results.get("contexts", [])
            sources = results.get("sources", [])
            scores = results.get("scores", [])
            vector_contexts = contexts
            graph_contexts = []
            entities_found = []
        
        # Generate answer
        if contexts:
            messages = build_messages(question, contexts)
            answer = self.llm.generate(messages)
        else:
            answer = "No relevant information found."
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "scores": scores,
            "vector_contexts": vector_contexts,
            "graph_contexts": graph_contexts,
            "entities_found": entities_found,
            "retrieval_type": "hybrid" if use_hybrid else "vector"
        }
    
    def get_knowledge_graph_stats(self) -> dict:
        """
        Get knowledge graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return self.knowledge_graph.get_stats()
    
    def search_entities(
        self,
        entity_name: str,
        depth: int = 2
    ) -> dict:
        """
        Search for an entity in the knowledge graph.
        
        Args:
            entity_name: Entity to search for
            depth: Graph traversal depth
            
        Returns:
            Entity network information
        """
        from src.retrieval.graph_retriever import get_graph_retriever
        
        retriever = get_graph_retriever(self.knowledge_graph)
        return retriever.get_entity_network(entity_name, depth)
    
    def clear(self, clear_qdrant: bool = False) -> None:
        """
        Clear the knowledge graph and optionally Qdrant.
        
        Args:
            clear_qdrant: Whether to also clear Qdrant
        """
        self.knowledge_graph.clear()
        
        if clear_qdrant:
            # Recreate collection
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            client = QdrantClient(url="http://localhost:6333")
            try:
                client.delete_collection("docs")
            except:
                pass
            
            client.create_collection(
                collection_name="docs",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
    
    def close(self) -> None:
        """Close all connections."""
        self.knowledge_graph.close()


# Default pipeline instance
_default_pipeline: Optional[GraphRAGPipeline] = None


def get_graph_rag_pipeline(
    config_path: str = "config/defaults.yaml"
) -> GraphRAGPipeline:
    """
    Get or create the default Graph RAG pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        GraphRAGPipeline instance
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = GraphRAGPipeline(config_path)
    return _default_pipeline
