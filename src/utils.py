import yaml
import os
import logging
import sys
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

_logging_configured = False
_model_cache: dict = {}


def embed_texts_with_model(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Embed a list of texts using a SentenceTransformer model.

    Models are cached in-process so repeated calls with the same model
    name do not reload weights.

    Args:
        texts: Strings to embed.
        model_name: HuggingFace model name (e.g. "allenai/specter2_base").
        batch_size: Encoding batch size.

    Returns:
        List of embedding vectors (one per input text).
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence_transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    model = _model_cache[model_name]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [list(map(float, v)) for v in embeddings]


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger under the 'rag' hierarchy.
    Configures a stdout StreamHandler on the root 'rag' logger once.
    Log level is read from config/defaults.yaml (logging.level), defaulting to INFO.
    """
    global _logging_configured
    if not _logging_configured:
        try:
            cfg = load_config()
            level_str = cfg.get("logging", {}).get("level", "INFO").upper()
        except Exception:
            level_str = "INFO"

        level = getattr(logging, level_str, logging.INFO)
        root = logging.getLogger("rag")
        root.setLevel(level)

        if not root.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            root.addHandler(handler)

        # Prevent log records from propagating to the root logger,
        # which may have handlers added by third-party libraries (transformers, etc.)
        root.propagate = False

        _logging_configured = True

    return logging.getLogger(f"rag.{name}")


def load_config(config_path: str = "config/defaults.yaml") -> dict:
    """Load the configuration from a YAML file."""
    if not os.path.exists(config_path):
        # Fallback to current directory or relative to root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "defaults.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_chunk_settings() -> tuple[int, int]:
    """Get chunk size and overlap from config."""
    config = load_config()
    chunk_size = config.get("chunker", {}).get("chunk_size", 1000)
    chunk_overlap = config.get("chunker", {}).get("chunk_overlap", 200)
    return chunk_size, chunk_overlap
