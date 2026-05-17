"""
ai/tools/risk_scorer_tool.py

LangChain tools for risk detection and scoring.

These tools are used by the Risk Agent to:
  - search for specific risk patterns in a document
  - check for missing standard clauses
  - flag ambiguous or one-sided language
  - calculate a weighted risk score from a list of flags
"""

import json
import logging
from typing import Optional

from langchain.tools import tool

from ai.rag.vector_store import similarity_search
from ai.schemas.risk_schema import RiskLevel

logger = logging.getLogger(__name__)

# Standard clauses every contract should have
STANDARD_CLAUSES = [
    "limitation of liability",
    "indemnification",
    "governing law",
    "dispute resolution",
    "confidentiality",
    "termination",
    "force majeure",
    "payment terms",
]

# Risk weights by category — used in score calculation
RISK_WEIGHTS = {
    "critical": 25,
    "high":     15,
    "medium":    8,
    "low":       3,
}


@tool
def check_missing_clauses(document_id: str) -> str:
    """
    Check which standard legal clauses are missing from a contract.
    Searches the vector store for each standard clause type.

    Input: document_id string
    Returns: JSON string listing present and missing standard clauses.
    """
    present = []
    missing = []

    for clause_name in STANDARD_CLAUSES:
        results = similarity_search(clause_name, document_id=document_id, k=1)

        if results and len(results[0].page_content.strip()) > 30:
            present.append(clause_name)
        else:
            missing.append(clause_name)

    logger.info(
        f"check_missing_clauses: doc={document_id} "
        f"present={len(present)} missing={len(missing)}"
    )
    return json.dumps({
        "document_id": document_id,
        "present":     present,
        "missing":     missing,
        "missing_count": len(missing),
    })


@tool
def search_risky_patterns(document_id: str) -> str:
    """
    Search a document for common high-risk legal patterns such as
    unilateral amendment rights, unlimited liability, vague termination,
    automatic renewal traps, and non-compete clauses.

    Input: document_id string
    Returns: JSON string with found risky patterns and their context.
    """
    risky_patterns = [
        ("unilateral amendment",     "high",   "one-sided",  "Party can change terms without consent"),
        ("unlimited liability",       "critical","liability",  "No cap on damages"),
        ("sole discretion",           "medium",  "ambiguity",  "Decision making without objective criteria"),
        ("automatically renew",       "medium",  "ambiguity",  "Automatic renewal may trap parties"),
        ("non-compete non-solicitation", "high", "punitive",  "Restrictive covenant may be unenforceable"),
        ("liquidated damages",        "high",    "punitive",   "Pre-set damages may be excessive"),
        ("waive right to jury",       "high",    "one-sided",  "Jury trial waiver disadvantages one party"),
        ("indemnify and hold harmless","medium",  "liability",  "Broad indemnification scope"),
        ("as is without warranty",    "high",    "one-sided",  "No warranty protection for buyer"),
        ("force majeure",             "low",     "compliance", "Check if force majeure scope is appropriate"),
    ]

    found_patterns = []

    for phrase, risk_level, category, description in risky_patterns:
        results = similarity_search(phrase, document_id=document_id, k=1)

        if results:
            chunk_text = results[0].page_content
            if any(word in chunk_text.lower() for word in phrase.split()):
                found_patterns.append({
                    "pattern":     phrase,
                    "risk_level":  risk_level,
                    "category":    category,
                    "description": description,
                    "context":     chunk_text[:300],
                    "page":        results[0].metadata.get("page"),
                })

    logger.info(
        f"search_risky_patterns: doc={document_id} "
        f"patterns_found={len(found_patterns)}"
    )
    return json.dumps({
        "document_id":   document_id,
        "patterns_found": found_patterns,
        "total":         len(found_patterns),
    })


@tool
def check_ambiguous_language(document_id: str) -> str:
    """
    Search a document for vague or ambiguous legal language that could
    be interpreted in multiple ways, increasing legal risk.

    Input: document_id string
    Returns: JSON string with ambiguous phrases and their context.
    """
    ambiguous_phrases = [
        "reasonable efforts",
        "best efforts",
        "commercially reasonable",
        "promptly",
        "without undue delay",
        "as soon as practicable",
        "material breach",
        "reasonable notice",
        "fair market value",
        "reasonable person",
    ]

    found = []

    for phrase in ambiguous_phrases:
        results = similarity_search(phrase, document_id=document_id, k=1)
        if results and phrase.lower() in results[0].page_content.lower():
            found.append({
                "phrase":  phrase,
                "context": results[0].page_content[:250],
                "page":    results[0].metadata.get("page"),
                "note":    f"'{phrase}' is subjective and may lead to disputes",
            })

    logger.info(
        f"check_ambiguous_language: doc={document_id} "
        f"ambiguous_found={len(found)}"
    )
    return json.dumps({
        "document_id":  document_id,
        "ambiguous":    found,
        "total":        len(found),
    })


@tool
def calculate_risk_score(flags_json: str) -> str:
    """
    Calculate a numeric risk score (0-100) from a list of risk flags.
    Uses weighted scoring based on risk level.

    Input: JSON string with a list of flags, each having a risk_level field.
    Example: '{"flags": [{"risk_level": "high"}, {"risk_level": "medium"}]}'

    Returns: JSON string with risk_score and overall_risk level.
    """
    try:
        data = json.loads(flags_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Input must be valid JSON"})

    flags = data.get("flags", [])

    if not flags:
        return json.dumps({
            "risk_score":   0,
            "overall_risk": "low",
            "breakdown":    {},
        })

    breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for flag in flags:
        level = flag.get("risk_level", "low").lower()
        if level in breakdown:
            breakdown[level] += 1

    raw_score = sum(
        count * RISK_WEIGHTS[level]
        for level, count in breakdown.items()
    )
    risk_score = min(100, raw_score)

    if breakdown["critical"] > 0 or risk_score >= 75:
        overall_risk = "critical"
    elif breakdown["high"] > 0 or risk_score >= 50:
        overall_risk = "high"
    elif breakdown["medium"] > 0 or risk_score >= 25:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    logger.info(
        f"calculate_risk_score: score={risk_score} "
        f"overall={overall_risk} flags={len(flags)}"
    )
    return json.dumps({
        "risk_score":   risk_score,
        "overall_risk": overall_risk,
        "total_flags":  len(flags),
        "breakdown":    breakdown,
    })


def get_all_risk_tools() -> list:
    """Return all risk scorer tools as a list for agent registration."""
    return [
        check_missing_clauses,
        search_risky_patterns,
        check_ambiguous_language,
        calculate_risk_score,
    ]