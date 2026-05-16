"""
ai/agents/extractor_agent.py

AGENT 1 — Extractor
Reads raw contract text → returns structured list of ExtractedClause objects.

Uses Mistral  with JSON-mode prompting.
Input : full contract text (string)
Output: ClauseExtractionResult (Pydantic)
"""

import json
import logging
from typing import List

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai.config import settings
from ai.prompts.agent_prompts import EXTRACTOR_SYSTEM, EXTRACTOR_HUMAN
from ai.schemas.clause_schema import (
    ExtractedClause,
    ClauseExtractionResult,
    ClauseType,
)

logger = logging.getLogger(__name__)

# Max chars sent to LLM in one shot — Mistral free has 32k context
MAX_TEXT_LENGTH = 28_000


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def _parse_response(raw_json: str, document_id: str, filename: str) -> ClauseExtractionResult:
    """Parse LLM JSON output into a ClauseExtractionResult."""
    try:
        # Strip markdown fences if the model added them despite instructions
        cleaned = raw_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Extractor JSON parse error: {e}\nRaw: {raw_json[:300]}")
        raise ValueError(f"Extractor agent returned invalid JSON: {e}")

    clauses: List[ExtractedClause] = []
    for item in data.get("clauses", []):
        # Safely coerce clause_type — default to OTHER if unrecognised
        try:
            ctype = ClauseType(item.get("clause_type", "other").lower())
        except ValueError:
            ctype = ClauseType.OTHER

        clauses.append(
            ExtractedClause(
                clause_id=item.get("clause_id", f"C-{len(clauses)+1:03d}"),
                clause_type=ctype,
                heading=item.get("heading"),
                text=item.get("text", ""),
                page=item.get("page"),
                section=item.get("section"),
                parties_mentioned=item.get("parties_mentioned", []),
            )
        )

    return ClauseExtractionResult(
        document_id=document_id,
        filename=filename,
        total_clauses=len(clauses),
        clauses=clauses,
    )


async def run_extractor(
    contract_text: str,
    document_id: str,
    filename: str,
) -> ClauseExtractionResult:
    """
    Run the Extractor Agent on a contract.

    Args:
        contract_text: full text of the contract (from document_loader)
        document_id:   unique doc ID
        filename:      original filename

    Returns:
        ClauseExtractionResult with all extracted clauses
    """
    # Truncate if too long for free-tier context window
    if len(contract_text) > MAX_TEXT_LENGTH:
        logger.warning(
            f"Contract text truncated from {len(contract_text)} to {MAX_TEXT_LENGTH} chars"
        )
        contract_text = contract_text[:MAX_TEXT_LENGTH]

    llm = _get_llm()

    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM),
        HumanMessage(content=EXTRACTOR_HUMAN.format(contract_text=contract_text)),
    ]

    logger.info(f"Extractor Agent running for document_id={document_id}")
    response = await llm.ainvoke(messages)
    raw = response.content

    result = _parse_response(raw, document_id, filename)
    logger.info(f"Extractor done — {result.total_clauses} clauses extracted")
    return result