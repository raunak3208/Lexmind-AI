"""
ai/agents/classifier_agent.py

AGENT 2 — Classifier
Takes extracted clauses → verifies / corrects their clause_type labels.

Uses Mistral 
Input : ClauseExtractionResult from extractor_agent
Output: Updated ClauseExtractionResult with corrected types
"""

import json
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai.config import settings
from ai.prompts.agent_prompts import CLASSIFIER_SYSTEM, CLASSIFIER_HUMAN
from ai.schemas.clause_schema import ClauseExtractionResult, ClauseType

logger = logging.getLogger(__name__)


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def _build_classifier_input(extraction: ClauseExtractionResult) -> str:
    """Convert extracted clauses to compact JSON for the classifier prompt."""
    items = [
        {
            "clause_id":   c.clause_id,
            "clause_type": c.clause_type.value,
            "heading":     c.heading,
            "text":        c.text[:400],   # send first 400 chars — enough to classify
        }
        for c in extraction.clauses
    ]
    return json.dumps(items, indent=2)


async def run_classifier(
    extraction: ClauseExtractionResult,
) -> ClauseExtractionResult:
    """
    Run the Classifier Agent — verify and correct clause types.

    Args:
        extraction: output from extractor_agent.run_extractor()

    Returns:
        Updated ClauseExtractionResult with corrected clause_type values
    """
    if not extraction.clauses:
        logger.warning("Classifier received empty clause list — skipping")
        return extraction

    llm = _get_llm()
    clauses_json = _build_classifier_input(extraction)

    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=CLASSIFIER_HUMAN.format(clauses_json=clauses_json)),
    ]

    logger.info(
        f"Classifier Agent running for document_id={extraction.document_id} "
        f"({len(extraction.clauses)} clauses)"
    )
    response = await llm.ainvoke(messages)
    raw = response.content

    # Parse corrections
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Classifier JSON parse error — keeping original types. Error: {e}")
        return extraction   # graceful fallback: return unchanged

    # Build a lookup: clause_id → corrected type
    corrections: dict[str, str] = {}
    for item in data.get("clauses", []):
        cid  = item.get("clause_id")
        ctype = item.get("clause_type", "").lower()
        if cid and ctype:
            corrections[cid] = ctype

    # Apply corrections in-place
    reclassified = 0
    for clause in extraction.clauses:
        new_type_str = corrections.get(clause.clause_id)
        if new_type_str:
            try:
                new_type = ClauseType(new_type_str)
                if new_type != clause.clause_type:
                    logger.debug(
                        f"Reclassified {clause.clause_id}: "
                        f"{clause.clause_type} → {new_type}"
                    )
                    clause.clause_type = new_type
                    reclassified += 1
            except ValueError:
                pass  # ignore unrecognised type strings

    logger.info(f"Classifier done — {reclassified} clauses reclassified")
    return extraction