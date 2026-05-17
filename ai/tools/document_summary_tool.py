"""
ai/tools/document_summary_tool.py

LangChain tools for document-level summarisation tasks.

These tools are used by the Summarizer Agent to:
  - extract key dates from a document
  - identify governing law and jurisdiction
  - find payment amounts and schedules
  - extract notice periods and deadlines
"""

import json
import logging
import re
from typing import Optional

from langchain.tools import tool

from ai.rag.vector_store import similarity_search

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
]

CURRENCY_PATTERN = r"\$[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)\b"


@tool
def extract_key_dates(document_id: str) -> str:
    """
    Extract important dates from a contract — effective date, expiry,
    notice periods, payment due dates.

    Input: document_id string
    Returns: JSON string with found dates and their context.
    """
    date_queries = [
        "effective date agreement entered into",
        "expiry termination expiration date",
        "notice period days written",
        "payment due date deadline",
        "renewal date automatically renew",
    ]

    found_dates = []
    seen_texts: set[str] = set()

    for query in date_queries:
        results = similarity_search(query, document_id=document_id, k=1)
        if not results:
            continue

        chunk_text = results[0].page_content
        key = chunk_text[:80]
        if key in seen_texts:
            continue
        seen_texts.add(key)

        dates_in_chunk = []
        for pattern in DATE_PATTERNS:
            matches = re.findall(pattern, chunk_text, re.IGNORECASE)
            dates_in_chunk.extend(matches)

        if dates_in_chunk:
            found_dates.append({
                "context":      chunk_text[:300],
                "dates_found":  list(set(dates_in_chunk)),
                "query":        query,
                "page":         results[0].metadata.get("page"),
            })

    logger.info(
        f"extract_key_dates: doc={document_id} "
        f"date_contexts={len(found_dates)}"
    )
    return json.dumps({
        "document_id": document_id,
        "date_contexts": found_dates,
        "total_contexts": len(found_dates),
    })


@tool
def extract_governing_law(document_id: str) -> str:
    """
    Find the governing law and jurisdiction clause in a contract.

    Input: document_id string
    Returns: JSON string with governing law context.
    """
    queries = [
        "governing law jurisdiction shall be governed",
        "laws of the state of",
        "disputes resolved in the courts of",
    ]

    for query in queries:
        results = similarity_search(query, document_id=document_id, k=1)
        if results:
            text = results[0].page_content
            if any(kw in text.lower() for kw in ["govern", "jurisdiction", "laws of"]):
                logger.info(f"extract_governing_law: doc={document_id} found")
                return json.dumps({
                    "document_id":   document_id,
                    "found":         True,
                    "context":       text[:400],
                    "page":          results[0].metadata.get("page"),
                })

    logger.info(f"extract_governing_law: doc={document_id} not found")
    return json.dumps({
        "document_id": document_id,
        "found":       False,
        "context":     None,
    })


@tool
def extract_payment_terms(document_id: str) -> str:
    """
    Extract payment amounts, schedules, and terms from a contract.

    Input: document_id string
    Returns: JSON string with payment information found.
    """
    queries = [
        "payment amount fee compensation shall pay",
        "invoice due net days",
        "monthly annual retainer rate",
    ]

    found = []
    seen: set[str] = set()

    for query in queries:
        results = similarity_search(query, document_id=document_id, k=2)
        for r in results:
            key = r.page_content[:80]
            if key in seen:
                continue
            seen.add(key)

            text = r.page_content
            amounts = re.findall(CURRENCY_PATTERN, text)
            net_terms = re.findall(r"\bnet\s+\d+\b", text, re.IGNORECASE)

            if amounts or net_terms:
                found.append({
                    "context":   text[:300],
                    "amounts":   amounts,
                    "net_terms": net_terms,
                    "page":      r.metadata.get("page"),
                })

    logger.info(
        f"extract_payment_terms: doc={document_id} "
        f"payment_contexts={len(found)}"
    )
    return json.dumps({
        "document_id":      document_id,
        "payment_contexts": found,
        "total":            len(found),
    })


@tool
def extract_obligations(document_id: str) -> str:
    """
    Extract key obligations for each party — what each party must do
    or is prohibited from doing under the contract.

    Input: document_id string
    Returns: JSON string with obligation statements found.
    """
    obligation_queries = [
        "shall must agree to obligations responsibilities",
        "party shall not prohibited restricted",
        "duty to provide deliver perform",
        "responsible for liable to ensure",
    ]

    found = []
    seen: set[str] = set()

    for query in obligation_queries:
        results = similarity_search(query, document_id=document_id, k=2)
        for r in results:
            key = r.page_content[:80]
            if key in seen:
                continue
            seen.add(key)

            text = r.page_content
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            obligation_sentences = [
                s for s in sentences
                if any(kw in s.lower() for kw in ["shall", "must", "agrees to", "will not", "shall not"])
            ]

            if obligation_sentences:
                found.append({
                    "obligations": obligation_sentences[:4],
                    "page":        r.metadata.get("page"),
                    "context":     text[:200],
                })

    logger.info(
        f"extract_obligations: doc={document_id} "
        f"obligation_blocks={len(found)}"
    )
    return json.dumps({
        "document_id":       document_id,
        "obligation_blocks": found,
        "total":             len(found),
    })


def get_all_summary_tools() -> list:
    """Return all document summary tools as a list for agent registration."""
    return [
        extract_key_dates,
        extract_governing_law,
        extract_payment_terms,
        extract_obligations,
    ]