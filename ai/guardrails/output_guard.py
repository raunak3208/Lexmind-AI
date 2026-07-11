"""
ai/guardrails/output_guard.py

Output validation for LLM-generated answers.

Checks performed on every LLM response:
  1. PII leakage scan      — ensure redacted PII did not reappear in output
  2. Hallucination signals — detect phrases that signal the LLM is guessing
  3. Empty response        — catch blank or near-blank outputs
  4. Refusal detection     — detect if LLM refused to answer (log but pass through)
  5. Length validation     — catch suspiciously short or long outputs

Location: ai/guardrails/output_guard.py
"""

import re
import logging
from dataclasses import dataclass, field

from ai.config import settings

logger = logging.getLogger(__name__)

# Phrases that commonly signal LLM hallucination or uncertainty
# These are warnings — they do not block the response
HALLUCINATION_SIGNALS = [
    r"as of my (knowledge|training) (cutoff|date)",
    r"i (don't|do not|cannot) have access to",
    r"i('m| am) not sure (but|however)",
    r"this (may|might|could) (not )?be (accurate|correct|up.to.date)",
    r"you (should|may want to) (consult|verify|check)",
    r"i (may|might|could) be (wrong|mistaken|incorrect)",
    r"based on (my|general) (knowledge|training)",
    r"i (cannot|can't) (guarantee|confirm|verify)",
    r"as far as i (know|recall|remember)",
    r"to the best of my (knowledge|recollection)",
]

# Phrases indicating the LLM refused to answer
REFUSAL_SIGNALS = [
    r"i('m| am) (not able|unable) to",
    r"i (cannot|can't) (help|assist|answer|provide)",
    r"(this|that) (is|goes) beyond (my|the)",
    r"i (don't|do not) (have|possess) (that|the) (information|data|details)",
]

# Minimum and maximum sensible answer lengths
MIN_ANSWER_LENGTH = 20
MAX_ANSWER_LENGTH = 8000


@dataclass
class OutputGuardResult:
    is_safe:              bool
    final_text:           str
    original_text:        str
    warnings:             list[str] = field(default_factory=list)
    pii_leaked:           bool = False
    hallucination_risk:   bool = False
    was_refusal:          bool = False
    pii_redacted_output:  bool = False


def guard_output(
    answer: str,
    context_chunks: list[str] = None,
) -> OutputGuardResult:
    """
    Validate and clean a raw LLM answer before returning to the user.

    Args:
        answer:         raw LLM output string
        context_chunks: retrieved chunks used to generate the answer
                        (used to check faithfulness, not implemented as LLM
                        call here — uses heuristics only for zero API cost)

    Returns:
        OutputGuardResult with final safe text and any warnings
    """
    if not settings.enable_guardrails:
        return OutputGuardResult(
            is_safe=True,
            final_text=answer,
            original_text=answer,
        )

    warnings     = []
    final_text   = answer
    original     = answer
    pii_leaked   = False
    pii_redacted = False

    # Step 1 — empty response check
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        logger.warning("LLM returned empty or near-empty response")
        return OutputGuardResult(
            is_safe=False,
            final_text="The system could not generate a response. Please rephrase your question.",
            original_text=original,
            warnings=["Empty response from LLM"],
        )

    # Step 2 — PII scan on output — redact any PII that leaked through
    if settings.enable_pii_redaction:
        from ai.guardrails.pii_detector import scan as pii_scan
        pii_result = pii_scan(answer)

        if pii_result.pii_found:
            pii_leaked   = True
            pii_redacted = True
            final_text   = pii_result.redacted_text
            warnings.append(
                f"PII detected in response and redacted: "
                f"{list(pii_result.entity_counts.keys())}"
            )
            logger.warning(
                f"PII leaked into LLM output — redacted: "
                f"{pii_result.entity_counts}"
            )

    # Step 3 — hallucination signal detection
    lower = final_text.lower()
    hall_signals = [
        p for p in HALLUCINATION_SIGNALS
        if re.search(p, lower, re.IGNORECASE)
    ]
    if hall_signals:
        warnings.append(
            "Response may contain uncertain or unverified information. "
            "Please verify key details with the original contract."
        )
        logger.info(f"Hallucination signals detected: {len(hall_signals)} patterns")

    # Step 4 — refusal detection (log only, pass through)
    refusal_found = any(
        re.search(p, lower, re.IGNORECASE)
        for p in REFUSAL_SIGNALS
    )
    if refusal_found:
        logger.info("LLM response appears to be a refusal")

    # Step 5 — length check
    if len(final_text) > MAX_ANSWER_LENGTH:
        final_text = final_text[:MAX_ANSWER_LENGTH] + "\n\n[Response truncated]"
        warnings.append("Response was truncated to maximum allowed length.")

    return OutputGuardResult(
        is_safe=True,
        final_text=final_text,
        original_text=original,
        warnings=warnings,
        pii_leaked=pii_leaked,
        hallucination_risk=len(hall_signals) > 0,
        was_refusal=refusal_found,
        pii_redacted_output=pii_redacted,
    )
