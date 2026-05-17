"""
ai/tools/clause_extractor_tool.py

LangChain tool that wraps clause extraction logic.
Can be given to any LangChain agent as a callable tool.

Also exposes standalone helpers used directly by the Risk Agent
and Summarizer Agent to query extracted clauses without going
through the full LLM extraction pipeline again.
"""

import json
import logging
from typing import Optional

from langchain.tools import tool
from langchain_core.documents import Document

from ai.rag.vector_store import similarity_search
from ai.schemas.clause_schema import ClauseType, ExtractedClause

logger = logging.getLogger(__name__)


@tool
def find_clauses_by_type(input_json: str) -> str:
    """
    Search the vector store for clauses of a specific type within a document.

    Input must be a JSON string with keys:
      - document_id (str)
      - clause_type (str) — e.g. payment, termination, liability
      - k (int, optional) — number of results, default 3

    Returns a JSON string with matching clause chunks.
    """
    try:
        data = json.loads(input_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Input must be valid JSON"})

    document_id = data.get("document_id")
    clause_type = data.get("clause_type", "").lower()
    k = int(data.get("k", 3))

    if not document_id or not clause_type:
        return json.dumps({"error": "document_id and clause_type are required"})

    try:
        ClauseType(clause_type)
    except ValueError:
        return json.dumps({"error": f"Unknown clause_type: {clause_type}"})

    query = f"{clause_type} clause terms and obligations"
    results = similarity_search(query, document_id=document_id, k=k)

    chunks = [
        {
            "text":        doc.page_content,
            "page":        doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "filename":    doc.metadata.get("filename"),
        }
        for doc in results
    ]

    logger.info(
        f"find_clauses_by_type: clause_type={clause_type} "
        f"doc={document_id} found={len(chunks)}"
    )
    return json.dumps({"clause_type": clause_type, "results": chunks})


@tool
def find_parties_in_document(document_id: str) -> str:
    """
    Search the vector store for text that mentions parties (signatories)
    in a contract. Returns relevant chunks likely to contain party names.

    Input: document_id string
    Returns: JSON string with text chunks
    """
    queries = [
        "parties to this agreement between",
        "entered into by and between",
        "hereinafter referred to as",
    ]

    all_results: list[Document] = []
    seen_indices: set[int] = set()

    for q in queries:
        results = similarity_search(q, document_id=document_id, k=2)
        for r in results:
            idx = r.metadata.get("chunk_index", -1)
            if idx not in seen_indices:
                seen_indices.add(idx)
                all_results.append(r)

    chunks = [
        {
            "text":        doc.page_content[:500],
            "page":        doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
        }
        for doc in all_results
    ]

    logger.info(f"find_parties_in_document: doc={document_id} chunks={len(chunks)}")
    return json.dumps({"document_id": document_id, "party_context": chunks})


@tool
def get_clause_text(input_json: str) -> str:
    """
    Retrieve a specific clause from the vector store by searching
    for its heading or a key phrase.

    Input must be a JSON string with keys:
      - document_id (str)
      - search_phrase (str) — heading or key phrase to search for
      - k (int, optional) — number of results, default 2

    Returns a JSON string with matching chunks.
    """
    try:
        data = json.loads(input_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Input must be valid JSON"})

    document_id = data.get("document_id")
    search_phrase = data.get("search_phrase", "")
    k = int(data.get("k", 2))

    if not document_id or not search_phrase:
        return json.dumps({"error": "document_id and search_phrase are required"})

    results = similarity_search(search_phrase, document_id=document_id, k=k)

    chunks = [
        {
            "text":        doc.page_content,
            "page":        doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
        }
        for doc in results
    ]

    logger.info(
        f"get_clause_text: phrase='{search_phrase[:40]}' "
        f"doc={document_id} found={len(chunks)}"
    )
    return json.dumps({"search_phrase": search_phrase, "results": chunks})


def get_all_clause_type_tools() -> list:
    """Return all clause extractor tools as a list for agent registration."""
    return [find_clauses_by_type, find_parties_in_document, get_clause_text]