"""
ai/guardrails/guardrail_service.py

Single entry point for all guardrail operations.
All routes and agents import from here — not from individual modules.

Functions:
  protect_query(query)             guard + sanitise user query
  protect_contract(text)           guard + sanitise contract text
  protect_chunks(chunks)           redact PII from chunk list before embedding
  protect_output(answer, contexts) guard + sanitise LLM output
  audit_log(event, details)        log guardrail events for audit trail

Location: ai/guardrails/guardrail_service.py
"""

import logging
from typing import Optional

from ai.config import settings

logger = logging.getLogger(__name__)


def protect_query(query: str) -> dict:
    """
    Guard a user query before it hits RAG or agents.

    Args:
        query: raw user input

    Returns:
        dict with keys:
          allowed       (bool)   — whether to proceed
          text          (str)    — sanitised query to use
          blocked_reason (str)   — why it was blocked (empty if allowed)
          warnings      (list)   — non-blocking warnings to include in response
    """
    if not settings.enable_guardrails:
        return {"allowed": True, "text": query, "blocked_reason": "", "warnings": []}

    from ai.guardrails.input_guard import guard_query
    result = guard_query(query)

    if not result.is_safe:
        logger.warning(f"Query blocked: {result.blocked_reason}")
        audit_log("QUERY_BLOCKED", {
            "reason":            result.blocked_reason,
            "injection_detected": result.injection_detected,
            "query_preview":     query[:80],
        })

    return {
        "allowed":        result.is_safe,
        "text":           result.sanitised_text,
        "blocked_reason": result.blocked_reason,
        "warnings":       result.warnings,
    }


def protect_contract(text: str) -> dict:
    """
    Guard and sanitise contract text before ingestion or analysis.

    Args:
        text: raw contract text from document loader

    Returns:
        dict with keys:
          allowed   (bool)  — whether to proceed
          text      (str)   — sanitised text to use
          warnings  (list)  — non-blocking warnings
    """
    if not settings.enable_guardrails:
        return {"allowed": True, "text": text, "warnings": []}

    from ai.guardrails.input_guard import guard_contract_text
    result = guard_contract_text(text)

    if result.warnings:
        audit_log("CONTRACT_WARNINGS", {"warnings": result.warnings})

    return {
        "allowed":  result.is_safe,
        "text":     result.sanitised_text,
        "warnings": result.warnings,
    }


def protect_chunks(chunks: list) -> list:
    """
    Redact PII from document chunks before embedding and storage.
    This ensures PII never enters the vector store.

    Args:
        chunks: list of LangChain Document objects

    Returns:
        same list with PII redacted from page_content
    """
    if not settings.enable_guardrails or not settings.enable_pii_redaction:
        return chunks

    from ai.guardrails.pii_detector import redact_chunk_list

    original_samples = [c.page_content[:50] for c in chunks[:3]]
    chunks = redact_chunk_list(chunks)
    redacted_samples = [c.page_content[:50] for c in chunks[:3]]

    # Log if redaction changed anything
    if original_samples != redacted_samples:
        audit_log("CHUNKS_REDACTED", {
            "total_chunks": len(chunks),
            "sample_before": original_samples[0] if original_samples else "",
            "sample_after":  redacted_samples[0] if redacted_samples else "",
        })

    return chunks


def protect_output(
    answer: str,
    context_chunks: Optional[list[str]] = None,
) -> dict:
    """
    Guard LLM output before returning to the user.

    Args:
        answer:         raw LLM answer
        context_chunks: retrieved context used (for future faithfulness checks)

    Returns:
        dict with keys:
          text        (str)   — safe final answer to show user
          warnings    (list)  — warnings to include in response
          pii_leaked  (bool)  — whether PII was found and redacted in output
          is_safe     (bool)  — whether output passed all checks
    """
    if not settings.enable_guardrails:
        return {
            "text":       answer,
            "warnings":   [],
            "pii_leaked": False,
            "is_safe":    True,
        }

    from ai.guardrails.output_guard import guard_output
    result = guard_output(answer, context_chunks)

    if result.pii_leaked:
        audit_log("OUTPUT_PII_REDACTED", {
            "pii_redacted": result.pii_redacted_output,
            "answer_preview": answer[:80],
        })

    if not result.is_safe:
        audit_log("OUTPUT_BLOCKED", {"original_preview": answer[:80]})

    return {
        "text":       result.final_text,
        "warnings":   result.warnings,
        "pii_leaked": result.pii_leaked,
        "is_safe":    result.is_safe,
    }


def scan_for_pii(text: str) -> dict:
    """
    Scan text and return a PII report without redacting.
    Used for audit endpoints and debugging.

    Args:
        text: text to scan

    Returns:
        dict with pii_found, entity_counts, findings list
    """
    if not settings.enable_pii_redaction:
        return {"pii_found": False, "entity_counts": {}, "findings": []}

    from ai.guardrails.pii_detector import scan
    result = scan(text)

    return {
        "pii_found":    result.pii_found,
        "entity_counts": result.entity_counts,
        "findings": [
            {
                "entity_type": f.entity_type,
                "text":        f.text,
                "start":       f.start,
                "end":         f.end,
                "score":       f.score,
            }
            for f in result.findings
        ],
    }


def audit_log(event: str, details: dict) -> None:
    """
    Structured audit log for guardrail events.
    In production this would write to a separate audit log file or database.
    Currently writes to the standard logger at WARNING level.

    Args:
        event:   event type string (QUERY_BLOCKED, OUTPUT_PII_REDACTED, etc.)
        details: dict of relevant details to log
    """
    logger.warning(f"[GUARDRAIL AUDIT] {event}: {details}")
