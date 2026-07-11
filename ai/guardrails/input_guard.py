"""
ai/guardrails/input_guard.py

Input validation for all incoming queries and contract text.

Checks performed on every user query:
  1. Length validation     — not empty, not too long
  2. Prompt injection      — detect attempts to override system prompt
  3. Off-topic detection   — flag queries unrelated to legal documents
  4. Encoding sanitisation — strip control characters, normalise whitespace

Checks performed on contract text before ingestion:
  1. Minimum content       — must have extractable text
  2. Language detection    — warn if not English
  3. Encoding sanitisation

Location: ai/guardrails/input_guard.py
"""

import re
import logging
from dataclasses import dataclass, field

from ai.config import settings

logger = logging.getLogger(__name__)

# Known prompt injection patterns — these attempt to override the system prompt
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(your\s+)?(previous|prior|system)\s+(prompt|instructions?)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+\w+",
    r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you\s+are",
    r"</?(system|user|assistant|human|ai)>",
    r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>",
    r"jailbreak",
    r"DAN\s+mode",
    r"pretend\s+(you|that\s+you)\s+(are|have\s+no)",
]

# Phrases that suggest the query is clearly off-topic for a legal platform
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|food|restaurant|movie|film|music|song|sport|game|joke)\b",
    r"\b(weather|forecast|temperature|climate)\b",
    r"\b(stock\s+price|crypto|bitcoin|trading|invest)\b",
    r"\b(write\s+code|debug|programming|python|javascript)\b",
]

# Legal domain keywords — if present, query is almost certainly on-topic
LEGAL_KEYWORDS = {
    "contract", "clause", "agreement", "termination", "payment",
    "liability", "indemnif", "confidential", "govern", "jurisdiction",
    "party", "parties", "obligation", "breach", "remedy", "damages",
    "warrant", "represent", "covenant", "enforce", "legal", "law",
    "court", "arbitrat", "dispute", "settle", "notice", "effective",
    "expir", "renew", "assign", "subcontract", "intellectual", "property",
    "force majeure", "default", "cure", "waiver", "amend", "exhibit",
}


@dataclass
class InputGuardResult:
    is_safe:           bool
    sanitised_text:    str
    blocked_reason:    str = ""
    warnings:          list[str] = field(default_factory=list)
    injection_detected: bool = False
    off_topic:         bool = False


def guard_query(query: str) -> InputGuardResult:
    """
    Validate and sanitise a user chat/search query.

    Args:
        query: raw user input string

    Returns:
        InputGuardResult with is_safe flag and sanitised text
    """
    if not settings.enable_guardrails:
        return InputGuardResult(is_safe=True, sanitised_text=query)

    warnings = []

    # Step 1 — sanitise encoding
    sanitised = _sanitise_text(query)

    # Step 2 — length checks
    if len(sanitised.strip()) < settings.min_query_length:
        return InputGuardResult(
            is_safe=False,
            sanitised_text=sanitised,
            blocked_reason="Query is too short. Please ask a complete question.",
        )

    if len(sanitised) > settings.max_query_length:
        logger.warning(f"Query truncated from {len(sanitised)} to {settings.max_query_length} chars")
        sanitised = sanitised[:settings.max_query_length]
        warnings.append("Query was truncated to maximum allowed length.")

    # Step 3 — prompt injection detection
    injection = _detect_injection(sanitised)
    if injection:
        logger.warning(f"Prompt injection attempt blocked: '{sanitised[:80]}'")
        return InputGuardResult(
            is_safe=False,
            sanitised_text=sanitised,
            blocked_reason="Query contains patterns that are not allowed.",
            injection_detected=True,
        )

    # Step 4 — off-topic detection (soft warning, not a block)
    off_topic = _detect_off_topic(sanitised)
    if off_topic:
        warnings.append(
            "This query may be off-topic for a legal document platform. "
            "For best results ask about contract clauses, terms, or obligations."
        )

    return InputGuardResult(
        is_safe=True,
        sanitised_text=sanitised,
        warnings=warnings,
        off_topic=off_topic,
    )


def guard_contract_text(text: str) -> InputGuardResult:
    """
    Validate contract text before ingestion or analysis.
    Less strict than query guard — mainly sanitises encoding
    and checks for minimum useful content.

    Args:
        text: raw contract text from document loader

    Returns:
        InputGuardResult — almost always safe, may have warnings
    """
    if not settings.enable_guardrails:
        return InputGuardResult(is_safe=True, sanitised_text=text)

    warnings = []
    sanitised = _sanitise_text(text)

    if len(sanitised.strip()) < 100:
        return InputGuardResult(
            is_safe=False,
            sanitised_text=sanitised,
            blocked_reason="Document contains insufficient text to analyse.",
        )

    # Warn if no legal keywords found — might be wrong document type
    lower = sanitised.lower()
    legal_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in lower)
    if legal_hits < 3:
        warnings.append(
            "Document does not appear to contain standard legal language. "
            "Results may be less accurate."
        )
        logger.info(f"Low legal keyword count ({legal_hits}) in document")

    return InputGuardResult(
        is_safe=True,
        sanitised_text=sanitised,
        warnings=warnings,
    )


def _sanitise_text(text: str) -> str:
    """
    Clean text for safe processing.
    - Remove null bytes and control characters
    - Normalise line endings
    - Collapse excessive whitespace
    - Strip leading/trailing whitespace
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse excessive spaces
    text = re.sub(r" {3,}", "  ", text)

    return text.strip()


def _detect_injection(text: str) -> bool:
    """
    Check if text contains prompt injection patterns.
    Returns True if injection detected.
    """
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


def _detect_off_topic(text: str) -> bool:
    """
    Check if text is clearly off-topic for a legal platform.
    Returns True only if off-topic pattern found AND no legal keywords present.
    """
    lower = text.lower()

    has_legal = any(kw in lower for kw in LEGAL_KEYWORDS)
    if has_legal:
        return False

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True

    return False
