"""Local dummy knowledge base for testing the knowledge-retrieval tool.

Nothing in the reviewed project sources (code, flow doc, architecture
diagrams, pricing doc) contained real GoodScore knowledge-base content —
no S3 bucket, no S3 Vectors index, no AgentCore Gateway config. This file
is a stand-in for that, used ONLY when KNOWLEDGE_BASE_MODE=local_dummy
(see config.py, knowledge_gateway.py) — it exists purely so the
knowledge-retrieval tool path can be exercised end-to-end on a developer
machine. It is NOT a semantic/vector search (that's what S3 Vectors would
provide in production) — it's a small, explicit keyword-overlap ranking
over a handful of realistic GoodScore/credit-education FAQ entries.

The actual sample data lives in dummy_knowledge_base.csv (id, title,
content, tags — tags ';'-separated), not in this file, so it can be
inspected or edited without touching code. This module only loads it and
implements the search.

Replace this module's role entirely with a real AgentCore Gateway call
(see knowledge_gateway.py) once Gateway URL, auth, and the real S3
Vectors-backed index are available — do not extend this file into a
production search implementation.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_CSV_PATH = Path(__file__).parent / "dummy_knowledge_base.csv"


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    tags: tuple[str, ...]


def _load_documents(csv_path: Path) -> tuple[KnowledgeDocument, ...]:
    """Read the dummy KB CSV into KnowledgeDocument rows.

    Raises FileNotFoundError with a clear message if the CSV is missing —
    this is a real external-file boundary (unlike the rest of this
    module, which is pure in-memory logic), so it gets an explicit,
    actionable error instead of silently returning an empty knowledge
    base that would make every search() call look like "no results".
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dummy knowledge base CSV not found at {csv_path}. "
            "It ships alongside dummy_knowledge_base.py — did it get moved or deleted?"
        )

    documents = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tags = tuple(t.strip() for t in row["tags"].split(";") if t.strip())
            documents.append(KnowledgeDocument(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                tags=tags,
            ))
    return tuple(documents)


_DOCUMENTS: tuple[KnowledgeDocument, ...] = _load_documents(_CSV_PATH)


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_PATTERN.findall(text)}


def search(query: str, top_k: int = 3) -> list[dict]:
    """Rank dummy documents by keyword overlap with the query.

    Deliberately simple (word-set intersection, no embeddings/vector
    index) — this only needs to exercise the tool-call path realistically
    for local testing, not to be a good search algorithm. See module
    docstring: this is not a stand-in for the real S3 Vectors semantic
    search production would use.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored: list[tuple[int, KnowledgeDocument]] = []
    for doc in _DOCUMENTS:
        doc_words = _tokenize(doc.title) | _tokenize(doc.content) | {t.lower() for t in doc.tags}
        overlap = len(query_words & doc_words)
        # Small bonus for tag matches — tags are curated topic labels, so a
        # hit there is a stronger signal than an incidental word in the body.
        tag_words = {t.lower() for t in doc.tags}
        overlap += len(query_words & tag_words)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"id": doc.id, "title": doc.title, "content": doc.content}
        for _, doc in scored[:top_k]
    ]
