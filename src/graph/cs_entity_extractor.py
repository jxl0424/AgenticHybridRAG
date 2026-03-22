"""
CS/AI domain entity extractor for arXiv papers.

Extracts entities (MODEL, DATASET, TASK, FRAMEWORK, etc.) using
rule-based dictionaries and regex patterns. Also parses _description
relationship sentences directly into structured (subject, rel, object) triples.

Exposes extract_entities(query) so it can be used as a drop-in replacement
for MedicalEntityExtractor inside GraphRetriever.
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from src.utils import get_logger

logger = get_logger(__name__)


CS_ENTITY_TYPES = {
    "PAPER": "arXiv papers and publications",
    "AUTHOR": "Researchers and authors",
    "MODEL": "Neural network models and architectures",
    "DATASET": "Benchmark datasets",
    "TASK": "NLP/CV/ML tasks",
    "METRIC": "Evaluation metrics",
    "ALGORITHM": "Algorithms and methods",
    "FRAMEWORK": "Software frameworks and libraries",
    "VENUE": "Conferences and journals",
}

CS_RELATIONSHIP_TYPES = {
    "PROPOSED_BY", "EVALUATED_ON", "ACHIEVES", "SOLVES",
    "PUBLISHED_IN", "USES", "EXTENDS", "COMPARED_TO", "INTRODUCED_IN",
}

# --- Dictionaries ---

FRAMEWORKS = {
    "pytorch", "tensorflow", "jax", "keras", "theano", "mxnet",
    "huggingface", "transformers", "langchain", "llama-index", "llamaindex",
    "spacy", "nltk", "sklearn", "scikit-learn", "xgboost", "lightgbm",
    "fastai", "ray", "dask", "spark", "openai", "anthropic",
    "sentence-transformers", "sentence_transformers", "qdrant", "neo4j",
    "faiss", "chroma", "weaviate", "pinecone",
}

TASKS = {
    "text classification", "named entity recognition", "ner",
    "question answering", "machine translation", "summarization",
    "sentiment analysis", "text generation", "language modeling",
    "image classification", "object detection", "image segmentation",
    "speech recognition", "natural language inference", "nli",
    "relation extraction", "coreference resolution",
    "semantic textual similarity", "sts", "information retrieval",
    "reading comprehension", "dialogue", "text-to-sql",
    "knowledge graph", "graph neural network", "gnn",
    "recommendation", "reinforcement learning", "rl",
    "retrieval-augmented generation", "rag", "in-context learning",
    "chain-of-thought", "few-shot learning", "zero-shot learning",
    "fine-tuning", "pretraining", "pre-training",
}

METRICS = {
    "bleu", "rouge", "rouge-1", "rouge-2", "rouge-l",
    "f1", "f1 score", "accuracy", "precision", "recall",
    "perplexity", "ppl", "ndcg", "map", "mrr",
    "exact match", "em", "bertscore", "meteor",
    "cer", "wer", "auc", "roc", "mse", "rmse", "mae",
    "hit rate", "hit@k", "recall@k", "precision@k",
}

VENUES = {
    "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl",
    "cvpr", "iccv", "eccv", "aaai", "ijcai", "sigir", "www",
    "kdd", "wsdm", "cikm", "recsys", "arxiv", "nature", "science",
    "tacl", "coling", "eacl", "findings of acl",
}

WELL_KNOWN_MODELS = {
    "bert", "roberta", "albert", "electra", "deberta",
    "gpt", "gpt-2", "gpt-3", "gpt-4", "gpt-4o", "chatgpt",
    "t5", "mt5", "bart", "pegasus", "led",
    "llama", "llama-2", "llama-3", "mistral", "mixtral",
    "claude", "gemini", "palm", "falcon", "bloom",
    "xlnet", "longformer", "bigbird", "reformer",
    "vit", "clip", "dall-e", "stable diffusion", "diffusion",
    "wav2vec", "whisper", "hubert",
    "word2vec", "glove", "fasttext",
    "transformer", "attention", "self-attention",
    "bm25", "dense passage retrieval", "dpr", "rag",
    "specter", "specter2", "scibert", "pubmedbert",
}

WELL_KNOWN_DATASETS = {
    "squad", "squad2", "triviaqa", "naturalquestions", "nq",
    "hotpotqa", "musique", "2wikimultihopqa",
    "ms marco", "msmarco", "beir", "miracl",
    "glue", "superglue", "snli", "mnli", "qqp", "sst",
    "imagenet", "coco", "voc", "cityscapes",
    "wikitext", "c4", "the pile", "openwebtext",
    "mmlu", "hellaswag", "winogrande", "arc", "truthfulqa",
    "humaneval", "mbpp", "gsm8k", "math",
    "cord-19", "pubmed", "mimic", "ehr",
}

# Compiled patterns
_ARXIV_ID_PATTERN = re.compile(r"\barXiv[:\s]?(\d{4}\.\d{4,5})\b", re.IGNORECASE)
_AUTHOR_ET_AL_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*) et al\.?", re.UNICODE
)
_CAPITALIZED_ACRONYM_PATTERN = re.compile(r"\b([A-Z][A-Z0-9\-]{1,11}[A-Z0-9])\b")
# Matches mixed-case hyphenated abbreviations like "CP-nets", "k-means", "t-SNE"
_HYPHENATED_ABBREV_PATTERN = re.compile(r"\b([A-Z]{1,4}-[a-z][a-zA-Z0-9\-]{2,})\b")

# Predicate -> relationship type mapping (longest match first)
_PREDICATE_MAP = [
    (re.compile(r"\bintroduced\s+(?:in|at)\b", re.I), "INTRODUCED_IN"),
    (re.compile(r"\bpublished\s+(?:in|at)\b", re.I), "PUBLISHED_IN"),
    (re.compile(r"\bpresented\s+(?:at|in)\b", re.I), "PUBLISHED_IN"),
    (re.compile(r"\bproposed\s+by\b", re.I), "PROPOSED_BY"),
    (re.compile(r"\bintroduced\s+by\b", re.I), "PROPOSED_BY"),
    (re.compile(r"\bdeveloped\s+by\b", re.I), "PROPOSED_BY"),
    (re.compile(r"\bevaluated\s+on\b", re.I), "EVALUATED_ON"),
    (re.compile(r"\btested\s+on\b", re.I), "EVALUATED_ON"),
    (re.compile(r"\bbenchmarked\s+on\b", re.I), "EVALUATED_ON"),
    (re.compile(r"\bachieves?\b", re.I), "ACHIEVES"),
    (re.compile(r"\boutperforms?\b", re.I), "ACHIEVES"),
    (re.compile(r"\bsolves?\b", re.I), "SOLVES"),
    (re.compile(r"\baddresses?\b", re.I), "SOLVES"),
    (re.compile(r"\bapplied\s+to\b", re.I), "SOLVES"),
    (re.compile(r"\buses?\b", re.I), "USES"),
    (re.compile(r"\bbased\s+on\b", re.I), "USES"),
    (re.compile(r"\bbuilt\s+on\b", re.I), "USES"),
    (re.compile(r"\bextends?\b", re.I), "EXTENDS"),
    (re.compile(r"\bbuilds?\s+upon\b", re.I), "EXTENDS"),
    (re.compile(r"\bcompared\s+to\b", re.I), "COMPARED_TO"),
    (re.compile(r"\bversus\b", re.I), "COMPARED_TO"),
    (re.compile(r"\bvs\.?\b", re.I), "COMPARED_TO"),
]


@dataclass
class ExtractedEntity:
    """A single extracted CS/AI entity."""
    text: str
    entity_type: str
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Result of CS entity extraction."""
    chunk_id: str
    text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)


class CSEntityExtractor:
    """
    Rule-based CS/AI entity extractor for arXiv papers.

    Combines dictionary lookups, regex patterns, and direct parsing of
    _description relationship sentences into structured triples.

    Exposes extract_entities(query) so it is duck-type compatible with
    MedicalEntityExtractor for use inside GraphRetriever.
    """

    def __init__(self):
        self._build_lookup_patterns()

    def _build_lookup_patterns(self):
        """Pre-compile dictionary lookup patterns for fast matching."""
        def _dict_pattern(terms: set[str]) -> re.Pattern:
            sorted_terms = sorted(terms, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_terms]
            return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

        self._framework_pat = _dict_pattern(FRAMEWORKS)
        self._task_pat = _dict_pattern(TASKS)
        self._metric_pat = _dict_pattern(METRICS)
        self._venue_pat = _dict_pattern(VENUES)
        self._model_pat = _dict_pattern(WELL_KNOWN_MODELS)
        self._dataset_pat = _dict_pattern(WELL_KNOWN_DATASETS)

    def extract(
        self, paragraph: str, description: str, chunk_id: str
    ) -> ExtractionResult:
        """
        Extract entities and relationships from paragraph text and
        a pre-structured description sentence.
        """
        entities: list[ExtractedEntity] = []
        relationships: list[tuple[str, str, str]] = []

        # Extract entities from the paragraph
        if paragraph:
            entities.extend(self._extract_from_text(paragraph))

        # Parse the description as a structured relationship sentence
        if description:
            desc_entities = self._extract_from_text(description)
            entities.extend(desc_entities)
            rels = self._parse_description_relation(description)
            relationships.extend(rels)

        # Deduplicate entities by (text.lower(), entity_type)
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return ExtractionResult(
            chunk_id=chunk_id,
            text=paragraph,
            entities=unique,
            relationships=relationships,
        )

    def extract_entities(self, query: str) -> ExtractionResult:
        """
        Extract entities from a plain query string.
        Compatible with GraphRetriever's duck-typed call:
            result = extractor.extract_entities(query)
            for e in result.entities: e.text
        """
        return self.extract(query, "", "query")

    def _extract_from_text(self, text: str) -> list[ExtractedEntity]:
        entities = []

        for m in self._framework_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="FRAMEWORK",
                start_pos=m.start(), end_pos=m.end(), confidence=0.95,
            ))

        for m in self._task_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="TASK",
                start_pos=m.start(), end_pos=m.end(), confidence=0.9,
            ))

        for m in self._metric_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="METRIC",
                start_pos=m.start(), end_pos=m.end(), confidence=0.9,
            ))

        for m in self._venue_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group().upper(), entity_type="VENUE",
                start_pos=m.start(), end_pos=m.end(), confidence=0.9,
            ))

        for m in self._model_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="MODEL",
                start_pos=m.start(), end_pos=m.end(), confidence=0.95,
            ))

        for m in self._dataset_pat.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="DATASET",
                start_pos=m.start(), end_pos=m.end(), confidence=0.9,
            ))

        # arXiv paper IDs
        for m in _ARXIV_ID_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                text=f"arXiv:{m.group(1)}", entity_type="PAPER",
                start_pos=m.start(), end_pos=m.end(), confidence=1.0,
            ))

        # "Author et al." patterns
        for m in _AUTHOR_ET_AL_PATTERN.finditer(text):
            author = m.group(1).strip()
            if len(author) > 2:
                entities.append(ExtractedEntity(
                    text=author, entity_type="AUTHOR",
                    start_pos=m.start(), end_pos=m.end(), confidence=0.8,
                ))

        # Capitalized acronyms not already captured (potential unknown models/datasets)
        known_spans = {(e.start_pos, e.end_pos) for e in entities}
        for m in _CAPITALIZED_ACRONYM_PATTERN.finditer(text):
            if (m.start(), m.end()) not in known_spans:
                acronym = m.group(1)
                # Skip single-word stop words and very short ones
                if len(acronym) >= 3 and acronym not in {"THE", "AND", "FOR", "NOT", "BUT", "ALL"}:
                    entities.append(ExtractedEntity(
                        text=acronym, entity_type="MODEL",
                        start_pos=m.start(), end_pos=m.end(), confidence=0.5,
                    ))
                    known_spans.add((m.start(), m.end()))

        # Mixed-case hyphenated abbreviations: CP-nets, t-SNE, k-means, VC-dimension
        for m in _HYPHENATED_ABBREV_PATTERN.finditer(text):
            if (m.start(), m.end()) not in known_spans:
                entities.append(ExtractedEntity(
                    text=m.group(1), entity_type="MODEL",
                    start_pos=m.start(), end_pos=m.end(), confidence=0.6,
                ))

        return entities

    def _parse_description_relation(
        self, description: str
    ) -> list[tuple[str, str, str]]:
        """
        Parse a _description sentence into (subject, rel_type, object) triples.

        Example: "BERT achieves state-of-the-art on SQuAD"
                 -> ("BERT", "ACHIEVES", "SQuAD")
        """
        relationships = []

        for pattern, rel_type in _PREDICATE_MAP:
            m = pattern.search(description)
            if m:
                subject = description[: m.start()].strip().rstrip(",.")
                obj = description[m.end() :].strip().lstrip(",.")
                # Clean up: take only the first noun phrase (up to comma or period)
                subject = re.split(r"[,.]", subject)[-1].strip()
                obj = re.split(r"[,.]", obj)[0].strip()
                if subject and obj and subject != obj:
                    relationships.append((subject, rel_type, obj))
                break  # one relationship per description sentence

        return relationships


# Module-level singleton
_default_extractor: Optional[CSEntityExtractor] = None


def get_cs_entity_extractor() -> CSEntityExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = CSEntityExtractor()
    return _default_extractor
