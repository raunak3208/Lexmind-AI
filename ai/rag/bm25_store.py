"""
ai/rag/bm25_store.py

BM25 sparse index for the hybrid search pipeline.

Responsibilities:
  - Tokenise and index document chunks on ingestion
  - Persist index to disk (data/bm25_index/) so it survives restarts
  - Search index by keyword query, return ranked Document objects
  - Delete document entries when a contract is deleted
  - Rebuild index from scratch if needed

BM25Okapi is the most widely used BM25 variant.
  k1=1.5  — term frequency saturation (default 1.5 works well for legal text)
  b=0.75  — length normalisation (penalises very long chunks slightly)

Tokenisation for legal text:
  - Lowercase
  - Split on whitespace and punctuation
  - Keep section numbers like "4.2" intact
  - Keep defined terms like "EffectiveDate" as-is

Location: ai/rag/bm25_store.py
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from ai.config import settings

logger = logging.getLogger(__name__)

INDEX_FILE     = "bm25_index.pkl"
METADATA_FILE  = "bm25_metadata.json"


def _get_index_dir() -> Path:
    path = Path(settings.bm25_index_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tokenise(text: str) -> list[str]:
    """
    Legal-aware tokeniser.
    Preserves section numbers (4.2), defined terms (EffectiveDate),
    and currency amounts ($5,000) which are important in legal text.
    """
    text = text.lower()
    # Split on whitespace and most punctuation, but keep . inside numbers
    tokens = re.findall(r'\b[\w][\w.]*[\w]\b|\b\w\b', text)
    # Remove pure stopword noise but keep legal keywords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
        'in', 'on', 'at', 'by', 'for', 'with', 'or', 'and', 'but',
        'not', 'this', 'that', 'these', 'those', 'it', 'its'
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class BM25Store:
    """
    Manages a BM25 index that mirrors the Chroma vector store.
    One entry per chunk, keyed by the same chunk metadata.
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._documents: list[Document] = []
        self._load()

    def _index_path(self) -> Path:
        return _get_index_dir() / INDEX_FILE

    def _metadata_path(self) -> Path:
        return _get_index_dir() / METADATA_FILE

    def _load(self) -> None:
        """Load existing index from disk if available."""
        idx_path  = self._index_path()
        meta_path = self._metadata_path()

        if idx_path.exists() and meta_path.exists():
            try:
                with open(idx_path, "rb") as f:
                    self._index = pickle.load(f)
                with open(meta_path, "r") as f:
                    raw = json.load(f)
                    self._documents = [
                        Document(
                            page_content=d["page_content"],
                            metadata=d["metadata"],
                        )
                        for d in raw
                    ]
                logger.info(
                    f"BM25 index loaded: {len(self._documents)} chunks"
                )
            except Exception as e:
                logger.warning(f"BM25 index load failed, starting fresh: {e}")
                self._index     = None
                self._documents = []
        else:
            logger.info("No BM25 index found — will be built on first ingest")

    def _save(self) -> None:
        """Persist index and document metadata to disk."""
        try:
            with open(self._index_path(), "wb") as f:
                pickle.dump(self._index, f)
            raw = [
                {
                    "page_content": d.page_content,
                    "metadata":     d.metadata,
                }
                for d in self._documents
            ]
            with open(self._metadata_path(), "w") as f:
                json.dump(raw, f)
            logger.info(f"BM25 index saved: {len(self._documents)} chunks")
        except Exception as e:
            logger.error(f"BM25 index save failed: {e}")

    def _rebuild(self) -> None:
        """Rebuild BM25 index from current document list."""
        if not self._documents:
            self._index = None
            return
        tokenised = [_tokenise(doc.page_content) for doc in self._documents]
        self._index = BM25Okapi(tokenised, k1=1.5, b=0.75)

    def add_chunks(self, chunks: list[Document]) -> None:
        """
        Add new chunks to the BM25 index.
        Called by document_loader after ingestion — mirrors add_chunks in vector_store.py.

        Args:
            chunks: same list passed to vector_store.add_chunks()
        """
        if not chunks:
            return

        doc_id = chunks[0].metadata.get("document_id", "unknown")
        logger.info(f"BM25: indexing {len(chunks)} chunks for doc={doc_id}")

        self._documents.extend(chunks)
        self._rebuild()
        self._save()

    def delete_document(self, document_id: str) -> None:
        """
        Remove all chunks for a document from the BM25 index.
        Called when a contract is deleted — mirrors delete_document in vector_store.py.

        Args:
            document_id: MongoDB document ID
        """
        before = len(self._documents)
        self._documents = [
            d for d in self._documents
            if d.metadata.get("document_id") != document_id
        ]
        removed = before - len(self._documents)
        logger.info(f"BM25: removed {removed} chunks for doc={document_id}")
        self._rebuild()
        self._save()

    def search(
        self,
        query: str,
        k: int = None,
        document_id: Optional[str] = None,
    ) -> list[tuple[float, Document]]:
        """
        BM25 keyword search.

        Args:
            query:       search query string
            k:           number of results (default from config)
            document_id: optional filter to one document

        Returns:
            list of (score, Document) sorted by score descending
        """
        k = k or settings.retriever_k

        if self._index is None or not self._documents:
            logger.warning("BM25 index is empty — returning no results")
            return []

        tokens = _tokenise(query)
        if not tokens:
            return []

        # If document_id filter, work on subset
        if document_id:
            filtered_docs = [
                (i, d) for i, d in enumerate(self._documents)
                if d.metadata.get("document_id") == document_id
            ]
            if not filtered_docs:
                return []

            # Build temporary index on filtered subset
            temp_tokenised = [_tokenise(d.page_content) for _, d in filtered_docs]
            temp_index = BM25Okapi(temp_tokenised, k1=1.5, b=0.75)
            scores = temp_index.get_scores(tokens)
            doc_list = [d for _, d in filtered_docs]
        else:
            scores = self._index.get_scores(tokens)
            doc_list = self._documents

        # Sort by score descending, take top k
        scored = sorted(
            zip(scores, doc_list),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        # Filter out zero-score results
        results = [(float(s), d) for s, d in scored if s > 0]

        logger.info(
            f"BM25 search: query='{query[:50]}'  "
            f"results={len(results)}  doc_filter={document_id or 'ALL'}"
        )
        return results

    def get_stats(self) -> dict:
        """Return BM25 index stats."""
        return {
            "total_chunks": len(self._documents),
            "index_built":  self._index is not None,
            "index_path":   str(self._index_path()),
        }


# Singleton — one BM25Store instance shared across the app
_bm25_store: Optional[BM25Store] = None


def get_bm25_store() -> BM25Store:
    """Return singleton BM25Store."""
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store