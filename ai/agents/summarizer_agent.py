"""
ai/agents/summarizer_agent.py

AGENT 4 — Summarizer  (final in the pipeline)
Takes classified clauses + RiskReport → produces ContractSummary.

Uses Mistral 
Input : ClauseExtractionResult + RiskReport
Output: ContractSummary (Pydantic)
"""

import json
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai.config import settings
from ai.prompts.agent_prompts import SUMMARIZER_SYSTEM, SUMMARIZER_HUMAN
from ai.schemas.clause_schema import ClauseExtractionResult
from ai.schemas.risk_schema import RiskReport
from ai.schemas.document_schema import ContractSummary

logger = logging.getLogger(__name__)


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0.1,   # slight creativity for summaries
        max_retries=3,
    )


def _build_summarizer_input(extraction: ClauseExtractionResult, risk: RiskReport) -> tuple[str, str]:
    """Build compact JSON representations of clauses and risk for the summarizer prompt."""
    clauses_short = [
        {
            "clause_id":   c.clause_id,
            "clause_type": c.clause_type.value,
            "heading":     c.heading,
            "text":        c.text[:300],
        }
        for c in extraction.clauses
    ]

    risk_short = {
        "overall_risk": risk.overall_risk.value,
        "risk_score":   risk.risk_score,
        "summary":      risk.summary,
        "top_flags": [
            {
                "clause_id":   f.clause_id,
                "risk_level":  f.risk_level.value,
                "category":    f.category,
                "description": f.description,
            }
            for f in risk.flags[:5]   # top 5 flags only to save tokens
        ],
    }

    return json.dumps(clauses_short, indent=2), json.dumps(risk_short, indent=2)


def _parse_summary_response(
    raw_json: str,
    document_id: str,
    filename: str,
) -> ContractSummary:
    """Parse LLM JSON output into ContractSummary."""
    try:
        cleaned = raw_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Summarizer JSON parse error: {e}")
        return ContractSummary(
            document_id=document_id,
            filename=filename,
            contract_type="Unknown",
            parties=[],
            executive_summary="Summary could not be generated due to a parsing error.",
        )

    return ContractSummary(
        document_id=document_id,
        filename=filename,
        contract_type=data.get("contract_type", "Unknown"),
        parties=data.get("parties", []),
        effective_date=data.get("effective_date"),
        expiry_date=data.get("expiry_date"),
        governing_law=data.get("governing_law"),
        key_obligations=data.get("key_obligations", []),
        executive_summary=data.get("executive_summary", ""),
    )


async def run_summarizer(
    extraction: ClauseExtractionResult,
    risk: RiskReport,
) -> ContractSummary:
    """
    Run the Summarizer Agent — final step of the pipeline.

    Args:
        extraction: classified clauses
        risk:       risk report from risk_agent

    Returns:
        ContractSummary with executive summary, parties, dates, obligations
    """
    llm = _get_llm()
    clauses_json, risk_json = _build_summarizer_input(extraction, risk)

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM),
        HumanMessage(content=SUMMARIZER_HUMAN.format(
            clauses_json=clauses_json,
            risk_json=risk_json,
        )),
    ]

    logger.info(f"Summarizer Agent running for document_id={extraction.document_id}")
    response = await llm.ainvoke(messages)

    summary = _parse_summary_response(response.content, extraction.document_id, extraction.filename)
    logger.info(
        f"Summarizer done — contract_type='{summary.contract_type}'  "
        f"parties={summary.parties}"
    )
    return summary