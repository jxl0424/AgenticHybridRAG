"""
Text chunking utilities for document processing.
"""
from llama_index.core.node_parser import SentenceSplitter


class TextChunker:
    """Handles text chunking for document processing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    def chunk_documents(self, docs: list) -> list[str]:
        """Extract text from documents and split into chunks."""
        chunks = []
        for doc in docs:
            text = getattr(doc, "text", None)
            if text:
                chunks.extend(self.splitter.split_text(text))
        return chunks
    
    def chunk_text(self, text: str) -> list[str]:
        """Split a single text string into chunks."""
        return self.splitter.split_text(text)
