"""
PDF loading and embedding utilities.
"""
import os
import re
import hashlib
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from src.utils import get_chunk_settings, get_logger

load_dotenv()

logger = get_logger(__name__)

# Medical-specific embedding model: PubMedBert fine-tuned on MS-MARCO
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBED_DIM = 768  # PubMedBert-MS-MARCO outputs 768-dimensional embeddings

# Initialize the medical embedding model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading medical embedding model on {device}: {EMBED_MODEL}")
local_embed_model = SentenceTransformer(EMBED_MODEL, device=device)
logger.info(f"Medical embedding model loaded. Dimension: {EMBED_DIM}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using local model."""
    if not texts:
        return []
    embeddings = local_embed_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# Standardized chunking settings from config
chunk_size, chunk_overlap = get_chunk_settings()
splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID based on document ID and chunk index."""
    import uuid
    hash_input = f"{document_id}_chunk_{chunk_index}".encode("utf-8")
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    return str(uuid.UUID(hash_hex[:32]))


def load_and_chunk_pdf(path: str) -> list[str]:
    """Load a PDF file and split it into chunks optimized for medical retrieval."""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        # Normalize whitespace: collapse runs of spaces/tabs but preserve newlines
        # so that paragraph/list structure is not destroyed
        clean_text = re.sub(r'[ \t]+', ' ', t)          # collapse horizontal whitespace
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # at most two consecutive newlines
        clean_text = clean_text.strip()
        chunks.extend(splitter.split_text(clean_text))
    return chunks
