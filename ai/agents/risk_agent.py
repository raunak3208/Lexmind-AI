"""
ai/agents/risk_agent.py

AGENT 3 — Risk Scorer
Takes classified clauses → identifies risks and produces a RiskReport.

Uses Mistral .
Input : ClauseExtractionResult (post-classifier)
Output: RiskReport (Pydantic)
"""

import json
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai.config import settings
from ai.prompts.agent_prompts import RISK_SYSTEM, RISK_HUMAN
from ai.schemas.clause_schema import ClauseExtractionResult
from ai.schemas.risk_schema import RiskReport, RiskFlag, RiskLevel

logger = logging.getLogger(__name__)


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def _build_risk_input(extraction: ClauseExtractionResult) -> str:
    """Serialize clauses for the risk prompt — full text needed for accurate risk detection."""
    items = [
        {
            "clause_id":   c.clause_id,
            "clause_type": c.clause_type.value,
            "heading":     c.heading,
            "text":        c.text[:600],  # slightly more text for risk analysis
        }
        for c in extraction.clauses
    ]
    return json.dumps(items, indent=2)


def _parse_risk_response(
    raw_json: str,
    document_id: str,
    filename: str,
) -> RiskReport:
    """Parse LLM JSON output into a RiskReport."""
    try:
        cleaned = raw_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Risk Agent JSON parse error: {e}")
        # Return a safe fallback report rather than crashing
        return RiskReport(
            document_id=document_id,
            filename=filename,
            overall_risk=RiskLevel.LOW,
            risk_score=0,
            total_flags=0,
            flags=[],
            summary="Risk analysis could not be completed due to a parsing error.",
        )

    flags: list[RiskFlag] = []
    for item in data.get("flags", []):
        try:
            risk_level = RiskLevel(item.get("risk_level", "low").lower())
        except ValueError:
            risk_level = RiskLevel.LOW

        flags.append(
            RiskFlag(
                flag_id=item.get("flag_id", f"R-{len(flags)+1:03d}"),
                clause_id=item.get("clause_id", "unknown"),
                risk_level=risk_level,
                category=item.get("category", "other"),
                description=item.get("description", ""),
                suggestion=item.get("suggestion", ""),
                flagged_text=item.get("flagged_text"),
            )
        )

    try:
        overall_risk = RiskLevel(data.get("overall_risk", "low").lower())
    except ValueError:
        overall_risk = RiskLevel.LOW

    risk_score = max(0, min(100, int(data.get("risk_score", 0))))

    return RiskReport(
        document_id=document_id,
        filename=filename,
        overall_risk=overall_risk,
        risk_score=risk_score,
        total_flags=len(flags),
        flags=flags,
        summary=data.get("summary", "No summary available."),
    )


async def run_risk_agent(
    extraction: ClauseExtractionResult,
) -> RiskReport:
    """
    Run the Risk Agent on classified clauses.

    Args:
        extraction: output from classifier_agent (ClauseExtractionResult)

    Returns:
        RiskReport with risk score, flags, and summary
    """
    llm = _get_llm()
    clauses_json = _build_risk_input(extraction)

    messages = [
        SystemMessage(content=RISK_SYSTEM),
        HumanMessage(content=RISK_HUMAN.format(clauses_json=clauses_json)),
    ]

    logger.info(f"Risk Agent running for document_id={extraction.document_id}")
    response = await llm.ainvoke(messages)

    report = _parse_risk_response(response.content, extraction.document_id, extraction.filename)
    logger.info(
        f"Risk Agent done — score={report.risk_score}  "
        f"overall={report.overall_risk}  flags={report.total_flags}"
    )
    return report