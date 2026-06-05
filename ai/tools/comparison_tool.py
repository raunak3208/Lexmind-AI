"""
ai/tools/comparison_tool.py

Contract Comparison Tool
Compares two contracts side-by-side using vector similarity.

Given two document IDs, for each clause type it finds the most similar
chunks from both documents and returns a structured diff.

Uses: Chroma similarity_search (no extra API cost).
"""

import logging
from typing import Optional
from pydantic import BaseModel

from ai.rag.vector_store import similarity_search
from ai.schemas.clause_schema import ClauseType

logger = logging.getLogger(__name__)


class ClauseComparison(BaseModel):
    clause_type:   str
    doc_a_text:    Optional[str] = None
    doc_b_text:    Optional[str] = None
    doc_a_found:   bool = False
    doc_b_found:   bool = False
    similarity_note: str = ""


class ComparisonResult(BaseModel):
    document_id_a: str
    document_id_b: str
    filename_a:    str
    filename_b:    str
    comparisons:   list[ClauseComparison]
    summary:       str


# Clause types we check in every comparison
COMPARISON_CLAUSE_TYPES = [
    ClauseType.PAYMENT,
    ClauseType.TERMINATION,
    ClauseType.CONFIDENTIALITY,
    ClauseType.LIABILITY,
    ClauseType.INDEMNIFICATION,
    ClauseType.GOVERNING_LAW,
    ClauseType.DISPUTE_RESOLUTION,
    ClauseType.FORCE_MAJEURE,
    ClauseType.WARRANTY,
    ClauseType.PENALTY,
]


async def compare_contracts(
    document_id_a: str,
    filename_a: str,
    document_id_b: str,
    filename_b: str,
) -> ComparisonResult:
    """
    Compare two contracts clause-by-clause using semantic search.

    For each clause type, we query both document vectors with the clause type
    as the search query and surface the most relevant chunk from each.

    Args:
        document_id_a: first document ID
        filename_a:    first document filename (for display)
        document_id_b: second document ID
        filename_b:    second document filename (for display)

    Returns:
        ComparisonResult with side-by-side clause comparisons
    """
    logger.info(
        f"Comparing contracts: {document_id_a} ({filename_a}) "
        f"vs {document_id_b} ({filename_b})"
    )

    comparisons: list[ClauseComparison] = []
    missing_in_a = []
    missing_in_b = []

    for clause_type in COMPARISON_CLAUSE_TYPES:
        query = f"{clause_type.value} clause terms and conditions"

        # Search in document A
        results_a = similarity_search(query, document_id=document_id_a, k=1)
        # Search in document B
        results_b = similarity_search(query, document_id=document_id_b, k=1)

        doc_a_text = results_a[0].page_content if results_a else None
        doc_b_text = results_b[0].page_content if results_b else None

        # Determine similarity note
        if doc_a_text and doc_b_text:
            note = "Both contracts contain this clause — review differences above."
        elif doc_a_text and not doc_b_text:
            missing_in_b.append(clause_type.value)
            note = f"⚠️ Only present in {filename_a} — missing from {filename_b}."
        elif doc_b_text and not doc_a_text:
            missing_in_a.append(clause_type.value)
            note = f"Only present in {filename_b} — missing from {filename_a}."
        else:
            note = " Not found in either contract."

        comparisons.append(
            ClauseComparison(
                clause_type=clause_type.value,
                doc_a_text=doc_a_text,
                doc_b_text=doc_b_text,
                doc_a_found=bool(doc_a_text),
                doc_b_found=bool(doc_b_text),
                similarity_note=note,
            )
        )

    # Build summary
    summary_parts = [
        f"Compared '{filename_a}' and '{filename_b}' across {len(COMPARISON_CLAUSE_TYPES)} clause types."
    ]
    if missing_in_b:
        summary_parts.append(
            f"'{filename_b}' is missing: {', '.join(missing_in_b)}."
        )
    if missing_in_a:
        summary_parts.append(
            f"'{filename_a}' is missing: {', '.join(missing_in_a)}."
        )
    if not missing_in_a and not missing_in_b:
        summary_parts.append("Both contracts appear to cover all standard clause types.")

    return ComparisonResult(
        document_id_a=document_id_a,
        document_id_b=document_id_b,
        filename_a=filename_a,
        filename_b=filename_b,
        comparisons=comparisons,
        summary=" ".join(summary_parts),
    )