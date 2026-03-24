"""
Local parquet-based loader for HybridRAG-Bench data.

Reads pre-downloaded parquet files from data/hybridrag/ and yields
typed records for ingestion into Neo4j and Qdrant.

Pure data access layer — zero DB knowledge.
"""
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.utils import get_logger

logger = get_logger(__name__)

HYBRIDRAG_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

DOMAINS = ["arxiv_ai", "arxiv_cy", "arxiv_qm"]

EXPECTED_FILES = (
    [f"kg/{d}/{f}.parquet" for d in DOMAINS for f in ("nodes", "edges", "edge_properties")]
    + [f"text_qa/{d}/papers.parquet" for d in DOMAINS]
)


def _sanitise_rel_type(rel_type: str) -> str:
    return "".join(c for c in rel_type if c.isalnum() or c == "_").upper() or "RELATED_TO"


# ---------------------------------------------------------------------------
# Typed record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeRecord:
    node_id: int
    domain: str
    display_name: str
    entity_type: str          # labels[1] from the parquet labels array
    embedding: list[float]    # pre-computed 768-dim SPECTER2 vector
    qdrant_id: uuid.UUID = field(init=False)

    def __post_init__(self):
        self.qdrant_id = uuid.uuid5(HYBRIDRAG_NS, f"{self.domain}:{self.node_id}")


@dataclass
class EdgeRecord:
    edge_id: int
    domain: str
    src_id: int
    dst_id: int
    rel_type: str             # already sanitised


@dataclass
class ChunkRecord:
    edge_id: int
    domain: str
    src_id: int
    dst_id: int
    rel_type: str             # sanitised — matches Neo4j exactly
    paragraph: str
    embedding: list[float]
    paper_id: str
    src_name: Optional[str] = None
    dst_name: Optional[str] = None
    qdrant_id: uuid.UUID = field(init=False)

    def __post_init__(self):
        self.qdrant_id = uuid.uuid5(HYBRIDRAG_NS, f"{self.domain}:{self.edge_id}")


@dataclass
class PaperRecord:
    arxiv_id: str
    domain: str
    title: str
    categories: str
    published: str
    md_text: str


@dataclass
class PaperChunkRecord:
    arxiv_id: str
    domain: str
    title: str
    chunk_index: int
    chunk_text: str
    qdrant_id: uuid.UUID = field(init=False)

    def __post_init__(self):
        self.qdrant_id = uuid.uuid5(HYBRIDRAG_NS, f"{self.arxiv_id}:{self.chunk_index}")


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _chunk_md_text(
    md_text: str,
    min_tokens: int = 128,
    max_tokens: int = 512,
) -> list[str]:
    """
    Split md_text on blank lines, merge short paragraphs forward until
    min_tokens is reached, hard cap at max_tokens.
    Token count approximated as whitespace-split word count.
    """
    paragraphs = [p.strip() for p in md_text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens
            if current_tokens >= min_tokens:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class LocalParquetLoader:
    """
    Reads HybridRAG-Bench parquet files from a local directory and yields
    typed records. Validates file existence and required columns before
    yielding anything.

    Usage:
        loader = LocalParquetLoader("data/hybridrag")
        loader.validate()
        for rec in loader.iter_nodes("arxiv_ai"):
            ...
    """

    def __init__(self, data_dir: str = "data/hybridrag"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check all 12 expected parquet files exist and have required columns.
        Raises RuntimeError listing all failures.
        """
        import pandas as pd

        errors: list[str] = []

        required_columns = {
            "nodes":            {"node_id", "display_name", "primary_label", "labels", "properties_json"},
            "edges":            {"edge_id", "src_id", "dst_id", "rel_type"},
            "edge_properties":  {"edge_id", "key", "value"},
            "papers":           {"arxiv_id", "title", "md_text"},
        }

        for rel_path in EXPECTED_FILES:
            fpath = self.data_dir / rel_path
            if not fpath.exists():
                errors.append(f"Missing: {fpath}")
                continue

            file_key = fpath.stem  # nodes / edges / edge_properties / papers
            if file_key in required_columns:
                try:
                    df = pd.read_parquet(fpath, columns=list(required_columns[file_key]))
                    missing = required_columns[file_key] - set(df.columns)
                    if missing:
                        errors.append(f"{fpath}: missing columns {missing}")
                except Exception as e:
                    errors.append(f"{fpath}: read error — {e}")

        if errors:
            raise RuntimeError("Validation failed:\n" + "\n".join(f"  {e}" for e in errors))

        logger.info("Validation passed: all 12 parquet files present with required columns")

    # ------------------------------------------------------------------
    # Node iteration
    # ------------------------------------------------------------------

    def iter_nodes(self, domain: str) -> Iterator[NodeRecord]:
        """Yield NodeRecords for _Embeddable rows only."""
        import pandas as pd

        path = self.data_dir / f"kg/{domain}/nodes.parquet"
        df = pd.read_parquet(path)
        df = df[df["primary_label"] == "_Embeddable"].reset_index(drop=True)
        logger.info(f"[{domain}] {len(df)} _Embeddable nodes")

        for _, row in df.iterrows():
            # labels is an array; entity_type is labels[1]
            _lv = row.get("labels")
            labels = _lv if _lv is not None else []
            if isinstance(labels, str):
                try:
                    labels = json.loads(labels)
                except Exception:
                    labels = [labels]
            entity_type = labels[1] if len(labels) > 1 else "Unknown"
            # Sanitise entity_type for use as a Neo4j label
            entity_type = "".join(c for c in entity_type if c.isalnum() or c == "_") or "Unknown"

            props = {}
            try:
                props = json.loads(row.get("properties_json") or "{}")
            except Exception:
                pass
            display_name = props.get("name") or row.get("display_name") or str(row["node_id"])

            embedding = props.get("_embedding")
            if embedding is None:
                continue  # no vector — skip
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except Exception:
                    continue

            yield NodeRecord(
                node_id=int(row["node_id"]),
                domain=domain,
                display_name=str(display_name),
                entity_type=entity_type,
                embedding=embedding,
            )

    # ------------------------------------------------------------------
    # Edge iteration
    # ------------------------------------------------------------------

    def iter_edges(self, domain: str) -> Iterator[EdgeRecord]:
        """Yield EdgeRecords with sanitised rel_type."""
        import pandas as pd

        path = self.data_dir / f"kg/{domain}/edges.parquet"
        df = pd.read_parquet(path)
        logger.info(f"[{domain}] {len(df)} edges")

        for _, row in df.iterrows():
            yield EdgeRecord(
                edge_id=int(row["edge_id"]),
                domain=domain,
                src_id=int(row["src_id"]),
                dst_id=int(row["dst_id"]),
                rel_type=_sanitise_rel_type(str(row["rel_type"])),
            )

    # ------------------------------------------------------------------
    # Chunk iteration
    # ------------------------------------------------------------------

    def iter_chunks(self, domain: str, include_src_names: bool = False) -> Iterator[ChunkRecord]:
        """
        Yield ChunkRecords by pivoting edge_properties.parquet and joining
        with edges.parquet on edge_id.
        """
        import pandas as pd

        ep_path = self.data_dir / f"kg/{domain}/edge_properties.parquet"
        ed_path = self.data_dir / f"kg/{domain}/edges.parquet"

        ep_df = pd.read_parquet(ep_path)
        ed_df = pd.read_parquet(ed_path)[["edge_id", "src_id", "dst_id", "rel_type"]]

        # Pivot: one row per edge_id, columns = key, value = value
        pivoted = ep_df.pivot_table(
            index="edge_id", columns="key", values="value", aggfunc="first"
        ).reset_index()

        # Join with edges to get src_id, dst_id, rel_type
        merged = pivoted.merge(ed_df, on="edge_id", how="left")

        # Build node_map for optional src/dst name lookup
        node_map: dict[int, str] = {}
        if include_src_names:
            nd_path = self.data_dir / f"kg/{domain}/nodes.parquet"
            nd_df = pd.read_parquet(nd_path, columns=["node_id", "display_name", "properties_json"])
            for _, nrow in nd_df.iterrows():
                props = {}
                try:
                    props = json.loads(nrow.get("properties_json") or "{}")
                except Exception:
                    pass
                name = props.get("name") or nrow.get("display_name") or ""
                node_map[int(nrow["node_id"])] = str(name)

        logger.info(f"[{domain}] {len(merged)} chunk records")

        for _, row in merged.iterrows():
            paragraph = str(row.get("_paragraph") or "").strip()
            if not paragraph:
                continue

            emb_raw = row.get("_embedding")
            if emb_raw is None:
                continue
            if isinstance(emb_raw, str):
                try:
                    embedding = json.loads(emb_raw)
                except Exception:
                    continue
            else:
                embedding = list(emb_raw)

            ref_raw = row.get("_ref") or ""
            paper_id = ""
            try:
                ref_data = json.loads(ref_raw)
                paper_id = str(ref_data.get("id") or ref_data.get("paper_id") or "")
            except Exception:
                paper_id = str(ref_raw)[:64]

            edge_id = int(row["edge_id"])
            src_id = int(row["src_id"]) if not pd.isna(row.get("src_id", float("nan"))) else 0
            dst_id = int(row["dst_id"]) if not pd.isna(row.get("dst_id", float("nan"))) else 0
            rel_type = _sanitise_rel_type(str(row.get("rel_type") or ""))

            yield ChunkRecord(
                edge_id=edge_id,
                domain=domain,
                src_id=src_id,
                dst_id=dst_id,
                rel_type=rel_type,
                paragraph=paragraph,
                embedding=embedding,
                paper_id=paper_id,
                src_name=node_map.get(src_id) if include_src_names else None,
                dst_name=node_map.get(dst_id) if include_src_names else None,
            )

    # ------------------------------------------------------------------
    # Paper iteration
    # ------------------------------------------------------------------

    def iter_papers(self, domain: str) -> Iterator[PaperRecord]:
        """Yield PaperRecords from papers.parquet."""
        import pandas as pd

        path = self.data_dir / f"text_qa/{domain}/papers.parquet"
        df = pd.read_parquet(path)
        logger.info(f"[{domain}] {len(df)} papers")

        for _, row in df.iterrows():
            metadata = {}
            try:
                metadata = json.loads(row.get("metadata") or "{}")
            except Exception:
                pass

            yield PaperRecord(
                arxiv_id=str(row.get("arxiv_id") or row.get("id") or ""),
                domain=domain,
                title=str(row.get("title") or ""),
                categories=str(metadata.get("categories") or row.get("categories") or ""),
                published=str(metadata.get("published") or row.get("published") or ""),
                md_text=str(row.get("md_text") or ""),
            )

    # ------------------------------------------------------------------
    # QA pair loading
    # ------------------------------------------------------------------

    def load_local_qa_pairs(
        self,
        domains: list[str] | None = None,
        split: str = "test",
        k_per_type: int = 5,
        seed: int | None = None,
        stratify: bool = True,
    ) -> list[dict]:
        """
        Load QA pairs from text_qa/{domain}/qa.parquet with stratified random sampling.

        Args:
            domains: Domains to load (default: all three arxiv_* domains).
            split: Dataset split to filter on ("test" or "train").
            k_per_type: Samples per question_type per run.
            seed: numpy RNG seed. If None, a random seed is chosen.
                  Seed printing is the caller's responsibility.
            stratify: If True, sample k_per_type per question_type.
                      If False, sample k_per_type * 6 from the full pool.

        Returns:
            Shuffled list of dicts with keys:
                question, ground_truth_answer, ground_truth_context, question_type, domain
        """
        import pandas as pd
        import numpy as np

        if seed is None:
            seed = int(np.random.default_rng().integers(0, 1_000_000))

        rng = np.random.default_rng(seed)

        if domains is None:
            domains = DOMAINS

        QUESTION_TYPES = [
            "single_hop", "single_hop_w_conditions", "multi_hop",
            "multi_hop_difficult", "open_ended", "counterfactual",
        ]

        all_dfs: list[pd.DataFrame] = []
        for domain in domains:
            path = self.data_dir / f"text_qa/{domain}/qa.parquet"
            if not path.exists():
                logger.warning(f"QA parquet not found, skipping: {path}")
                continue
            df = pd.read_parquet(path)
            if "split" in df.columns:
                df = df[df["split"] == split]
            df = df[df["question"].notna() & df["answer"].notna()]
            df = df[df["question"].astype(str).str.strip() != ""]
            df = df[df["answer"].astype(str).str.strip() != ""]
            df["domain"] = domain
            all_dfs.append(df)

        if not all_dfs:
            return []

        full_df = pd.concat(all_dfs, ignore_index=True)

        if not stratify:
            n = min(k_per_type * len(QUESTION_TYPES), len(full_df))
            sampled = full_df.sample(n=n, random_state=rng)
        else:
            sampled_groups: list[pd.DataFrame] = []
            for qt in QUESTION_TYPES:
                group_df = full_df[full_df["question_type"] == qt]
                if len(group_df) == 0:
                    continue
                if len(group_df) < k_per_type:
                    print(f"WARNING: only {len(group_df)} rows available for '{qt}', using all")
                    sampled_groups.append(group_df)
                else:
                    sampled_groups.append(
                        group_df.sample(n=k_per_type, random_state=rng)
                    )
            if not sampled_groups:
                return []
            sampled = pd.concat(sampled_groups, ignore_index=True)

        # Final shuffle using the same RNG instance (no reset)
        idx = rng.permutation(len(sampled))
        sampled = sampled.iloc[idx].reset_index(drop=True)

        pairs = []
        for _, row in sampled.iterrows():
            pairs.append({
                "question": str(row["question"]).strip(),
                "ground_truth_answer": str(row["answer"]).strip(),
                "ground_truth_context": "",
                "question_type": str(row.get("question_type") or ""),
                "domain": str(row.get("domain") or ""),
            })

        logger.info(
            f"Loaded {len(pairs)} QA pairs "
            f"(split={split}, seed={seed}, k_per_type={k_per_type}, stratify={stratify})"
        )
        return pairs

    def iter_paper_chunks(
        self,
        domain: str,
        min_tokens: int = 128,
        max_tokens: int = 512,
    ) -> Iterator[PaperChunkRecord]:
        """Yield PaperChunkRecords by chunking paper md_text."""
        for paper in self.iter_papers(domain):
            if not paper.md_text:
                continue
            chunks = _chunk_md_text(paper.md_text, min_tokens, max_tokens)
            for idx, chunk_text in enumerate(chunks):
                yield PaperChunkRecord(
                    arxiv_id=paper.arxiv_id,
                    domain=domain,
                    title=paper.title,
                    chunk_index=idx,
                    chunk_text=chunk_text,
                )
