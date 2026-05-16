"""
ai/agents/orchestrator.py

ORCHESTRATOR — runs the full multi-agent pipeline in order:

  Document Text
       │
       ▼
  [1] Extractor Agent   → ClauseExtractionResult
       │
       ▼
  [2] Classifier Agent  → ClauseExtractionResult (corrected types)
       │
       ▼
  [3] Risk Agent        → RiskReport
       │
       ▼
  [4] Summarizer Agent  → ContractSummary
       │
       ▼
  FullAnalysisResult  (returned to FastAPI → Node backend)

All agents use Mistral (free tier).
"""

import logging
import time

from ai.agents.extractor_agent   import run_extractor
from ai.agents.classifier_agent  import run_classifier
from ai.agents.risk_agent        import run_risk_agent
from ai.agents.summarizer_agent  import run_summarizer
from ai.schemas.document_schema  import FullAnalysisResult

logger = logging.getLogger(__name__)


async def run_full_pipeline(
    contract_text: str,
    document_id: str,
    filename: str,
) -> FullAnalysisResult:
    """
    Run the complete 4-agent analysis pipeline on a contract.

    Args:
        contract_text: full extracted text from the contract file
        document_id:   unique ID (assigned by Node/MongoDB)
        filename:      original filename for display

    Returns:
        FullAnalysisResult containing extraction, risk_report, and summary
    """
    start = time.time()
    logger.info(
        f"=== Pipeline START  document_id={document_id}  file={filename} "
        f"text_len={len(contract_text)} ==="
    )

    # ── Step 1: Extract clauses ───────────────────────────────────────────────
    logger.info("Step 1/4 — Extractor Agent")
    extraction = await run_extractor(
        contract_text=contract_text,
        document_id=document_id,
        filename=filename,
    )
    logger.info(f"  → {extraction.total_clauses} clauses extracted")

    # ── Step 2: Classify / correct clause types ───────────────────────────────
    logger.info("Step 2/4 — Classifier Agent")
    extraction = await run_classifier(extraction)

    # ── Step 3: Score risks ───────────────────────────────────────────────────
    logger.info("Step 3/4 — Risk Agent")
    risk_report = await run_risk_agent(extraction)
    logger.info(
        f"  → risk_score={risk_report.risk_score}  "
        f"overall={risk_report.overall_risk}  flags={risk_report.total_flags}"
    )

    # ── Step 4: Summarise ─────────────────────────────────────────────────────
    logger.info("Step 4/4 — Summarizer Agent")
    summary = await run_summarizer(extraction, risk_report)
    logger.info(f"  → contract_type='{summary.contract_type}'")

    elapsed = round(time.time() - start, 2)
    logger.info(f"=== Pipeline DONE  document_id={document_id}  elapsed={elapsed}s ===")

    return FullAnalysisResult(
        document_id=document_id,
        filename=filename,
        status="completed",
        extraction=extraction,
        risk_report=risk_report,
        summary=summary,
    )