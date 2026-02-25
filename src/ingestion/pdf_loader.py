"""
PDF loading and embedding utilities.
"""
import os

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Medical-specific embedding model: PubMedBert fine-tuned on MS-MARCO
# Trained on biomedical literature - optimized for clinical/medical text retrieval
# Drop-in replacement for all-mpnet-base-v2 (same 768-dim output)
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBED_DIM = 768  # PubMedBert-MS-MARCO outputs 768-dimensional embeddings

# Initialize the medical embedding model
print(f"Loading medical embedding model: {EMBED_MODEL}")
local_embed_model = SentenceTransformer(EMBED_MODEL)
print(f"Medical embedding model loaded. Dimension: {EMBED_DIM}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using local model."""
    embeddings = local_embed_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> list[str]:
    """Load a PDF file and split it into chunks."""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks
