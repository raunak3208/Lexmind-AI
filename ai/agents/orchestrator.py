"""
ai/agents/orchestrator.py

Runs the full multi-agent pipeline in order:

  Document Text
       |
       v
  [Pre-analysis Tools]  <- clause extractor, risk pattern, summary tools
       |
       v
  [1] Extractor Agent   -> ClauseExtractionResult
       |
       v
  [2] Classifier Agent  -> ClauseExtractionResult (corrected types)
       |
       v
  [3] Risk Agent        -> RiskReport  (augmented with tool findings)
       |
       v
  [4] Summarizer Agent  -> ContractSummary (augmented with tool findings)
       |
       v
  FullAnalysisResult
"""

import json
import logging
import time
from typing import Optional

from ai.agents.extractor_agent   import run_extractor
from ai.agents.classifier_agent  import run_classifier
from ai.agents.risk_agent        import run_risk_agent
from ai.agents.summarizer_agent  import run_summarizer
from ai.schemas.document_schema  import FullAnalysisResult
from ai.schemas.risk_schema      import RiskFlag, RiskLevel

logger = logging.getLogger(__name__)


async def _run_pre_analysis_tools(document_id: str) -> dict:
    """
    Run all non-LLM tools against the vector store before agents start.
    Returns a dict of findings that agents can use to augment their output.
    This is fast — no Mistral calls, pure vector search.
    """
    from ai.tools.risk_scorer_tool      import check_missing_clauses, search_risky_patterns, check_ambiguous_language
    from ai.tools.document_summary_tool import extract_key_dates, extract_governing_law, extract_payment_terms, extract_obligations

    findings = {}

    try:
        findings["missing_clauses"] = json.loads(check_missing_clauses.invoke(document_id))
        logger.info(
            f"Pre-analysis: missing_clauses={findings['missing_clauses'].get('missing_count', 0)}"
        )
    except Exception as e:
        logger.warning(f"check_missing_clauses failed: {e}")
        findings["missing_clauses"] = {}

    try:
        findings["risky_patterns"] = json.loads(search_risky_patterns.invoke(document_id))
        logger.info(
            f"Pre-analysis: risky_patterns={findings['risky_patterns'].get('total', 0)}"
        )
    except Exception as e:
        logger.warning(f"search_risky_patterns failed: {e}")
        findings["risky_patterns"] = {}

    try:
        findings["ambiguous_language"] = json.loads(check_ambiguous_language.invoke(document_id))
        logger.info(
            f"Pre-analysis: ambiguous={findings['ambiguous_language'].get('total', 0)}"
        )
    except Exception as e:
        logger.warning(f"check_ambiguous_language failed: {e}")
        findings["ambiguous_language"] = {}

    try:
        findings["key_dates"] = json.loads(extract_key_dates.invoke(document_id))
    except Exception as e:
        logger.warning(f"extract_key_dates failed: {e}")
        findings["key_dates"] = {}

    try:
        findings["governing_law"] = json.loads(extract_governing_law.invoke(document_id))
    except Exception as e:
        logger.warning(f"extract_governing_law failed: {e}")
        findings["governing_law"] = {}

    try:
        findings["payment_terms"] = json.loads(extract_payment_terms.invoke(document_id))
    except Exception as e:
        logger.warning(f"extract_payment_terms failed: {e}")
        findings["payment_terms"] = {}

    try:
        findings["obligations"] = json.loads(extract_obligations.invoke(document_id))
    except Exception as e:
        logger.warning(f"extract_obligations failed: {e}")
        findings["obligations"] = {}

    return findings


def _augment_risk_report(risk_report, tool_findings: dict):
    """
    Merge tool-discovered risks into the agent's RiskReport.
    Adds flags for missing clauses and risky patterns found by tools
    that the LLM agent may have missed.
    """
    existing_flag_count = len(risk_report.flags)
    new_flags = []
    flag_counter = existing_flag_count + 1

    # Missing clause flags
    missing = tool_findings.get("missing_clauses", {}).get("missing", [])
    for clause_name in missing:
        new_flags.append(
            RiskFlag(
                flag_id=f"R-T{flag_counter:03d}",
                clause_id="N/A",
                risk_level=RiskLevel.MEDIUM,
                category="missing-clause",
                description=f"Standard '{clause_name}' clause not found in this contract.",
                suggestion=f"Add a {clause_name} clause to protect both parties.",
                flagged_text=None,
            )
        )
        flag_counter += 1

    # Risky pattern flags from tool
    patterns = tool_findings.get("risky_patterns", {}).get("patterns_found", [])
    for pattern in patterns:
        try:
            risk_level = RiskLevel(pattern.get("risk_level", "medium"))
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        new_flags.append(
            RiskFlag(
                flag_id=f"R-T{flag_counter:03d}",
                clause_id="N/A",
                risk_level=risk_level,
                category=pattern.get("category", "other"),
                description=pattern.get("description", ""),
                suggestion=f"Review the '{pattern.get('pattern')}' language carefully.",
                flagged_text=pattern.get("context", "")[:200],
            )
        )
        flag_counter += 1

    if new_flags:
        risk_report.flags.extend(new_flags)
        risk_report.total_flags = len(risk_report.flags)

        # Recalculate score with new flags
        from ai.tools.risk_scorer_tool import calculate_risk_score
        flags_input = json.dumps({"flags": [{"risk_level": f.risk_level.value} for f in risk_report.flags]})
        score_result = json.loads(calculate_risk_score.invoke(flags_input))
        risk_report.risk_score = score_result.get("risk_score", risk_report.risk_score)
        try:
            risk_report.overall_risk = RiskLevel(score_result.get("overall_risk", risk_report.overall_risk.value))
        except ValueError:
            pass

        logger.info(
            f"Risk augmented with {len(new_flags)} tool flags. "
            f"New score={risk_report.risk_score} total_flags={risk_report.total_flags}"
        )

    return risk_report


async def run_full_pipeline(
    contract_text: str,
    document_id: str,
    filename: str,
) -> FullAnalysisResult:
    """
    Run the complete pipeline — tools first, then 4 agents.

    Args:
        contract_text: full text of the contract
        document_id:   unique ID from MongoDB
        filename:      original filename

    Returns:
        FullAnalysisResult
    """
    start = time.time()
    logger.info(
        f"Pipeline START document_id={document_id} "
        f"file={filename} text_len={len(contract_text)}"
    )

    # Pre-analysis tools (vector search only, no LLM cost)
    logger.info("Pre-analysis: running tools against vector store")
    tool_findings = await _run_pre_analysis_tools(document_id)

    # Step 1: Extract clauses
    logger.info("Step 1/4 — Extractor Agent")
    extraction = await run_extractor(
        contract_text=contract_text,
        document_id=document_id,
        filename=filename,
    )
    logger.info(f"  clauses={extraction.total_clauses}")

    # Step 2: Classify
    logger.info("Step 2/4 — Classifier Agent")
    extraction = await run_classifier(extraction)

    # Step 3: Risk scoring
    logger.info("Step 3/4 — Risk Agent")
    risk_report = await run_risk_agent(extraction)

    # Augment risk report with tool findings
    risk_report = _augment_risk_report(risk_report, tool_findings)
    logger.info(
        f"  risk_score={risk_report.risk_score} "
        f"overall={risk_report.overall_risk} "
        f"total_flags={risk_report.total_flags}"
    )

    # Step 4: Summarise
    logger.info("Step 4/4 — Summarizer Agent")
    summary = await run_summarizer(extraction, risk_report)

    # Patch governing law from tool if agent missed it
    tool_gov_law = tool_findings.get("governing_law", {})
    if not summary.governing_law and tool_gov_law.get("found"):
        context = tool_gov_law.get("context", "")
        if context:
            summary.governing_law = context[:100]

    elapsed = round(time.time() - start, 2)
    logger.info(f"Pipeline DONE document_id={document_id} elapsed={elapsed}s")

    return FullAnalysisResult(
        document_id=document_id,
        filename=filename,
        status="completed",
        extraction=extraction,
        risk_report=risk_report,
        summary=summary,
    )