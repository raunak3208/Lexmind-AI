"""
ai/guardrails/pii_detector.py

PII detection and redaction using Microsoft Presidio.
Runs 100% locally — no data leaves your machine.

What it detects:
  PERSON          — names (John Doe, Jane Smith)
  EMAIL_ADDRESS   — email addresses
  PHONE_NUMBER    — phone numbers in any format
  CREDIT_CARD     — credit card numbers
  IBAN_CODE       — bank account numbers
  IP_ADDRESS      — IP addresses
  LOCATION        — addresses, cities, countries
  DATE_TIME       — dates (can reveal contract timelines)
  NRP             — national registration / SSN
  MEDICAL_LICENSE — medical identifiers
  URL             — URLs

Two modes:
  redact(text)    — replace PII with entity type labels
                    "John Doe signed" → "[PERSON] signed"
  scan(text)      — detect PII without redacting, return findings report
                    Used for audit logging

Location: ai/guardrails/pii_detector.py
"""

import logging
from functools import lru_cache
from typing import Optional
from dataclasses import dataclass

from ai.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PIIFinding:
    entity_type: str
    text:        str
    start:       int
    end:         int
    score:       float


@dataclass
class PIIScanResult:
    original_text:  str
    redacted_text:  str
    findings:       list[PIIFinding]
    pii_found:      bool
    entity_counts:  dict[str, int]


@lru_cache(maxsize=1)
def _get_analyzer():
    """
    Load Presidio AnalyzerEngine once and cache it.
    Tries spaCy en_core_web_sm first.
    Falls back to transformers-based engine if spaCy model not installed.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    logger.info("Loading Presidio AnalyzerEngine...")

    try:
        import spacy
        spacy.load("en_core_web_sm")
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        logger.info("Presidio ready: spaCy backend")
    except (OSError, ImportError) as e:
        logger.warning(f"spaCy model not found ({e}) — using pattern-only Presidio")
        # Pattern-only mode — still catches emails, phones, credit cards, IPs
        analyzer = AnalyzerEngine()

    return analyzer


@lru_cache(maxsize=1)
def _get_anonymizer():
    """Load Presidio AnonymizerEngine once and cache it."""
    from presidio_anonymizer import AnonymizerEngine
    return AnonymizerEngine()


def scan(
    text: str,
    score_threshold: float = 0.6,
) -> PIIScanResult:
    """
    Scan text for PII without redacting.
    Returns a detailed report of what was found and where.

    Args:
        text:             text to scan
        score_threshold:  minimum confidence score to flag (0.0-1.0)

    Returns:
        PIIScanResult with findings list and entity counts
    """
    if not text or not settings.enable_pii_redaction:
        return PIIScanResult(
            original_text=text,
            redacted_text=text,
            findings=[],
            pii_found=False,
            entity_counts={},
        )

    try:
        analyzer = _get_analyzer()
        results  = analyzer.analyze(
            text=text,
            entities=settings.pii_entities,
            language="en",
            score_threshold=score_threshold,
        )

        findings = [
            PIIFinding(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=round(r.score, 3),
            )
            for r in results
        ]

        entity_counts: dict[str, int] = {}
        for f in findings:
            entity_counts[f.entity_type] = entity_counts.get(f.entity_type, 0) + 1

        if findings:
            logger.info(
                f"PII scan: {len(findings)} entities found — "
                f"{entity_counts}"
            )

        # Redact for the result
        redacted = _redact_text(text, results)

        return PIIScanResult(
            original_text=text,
            redacted_text=redacted,
            findings=findings,
            pii_found=len(findings) > 0,
            entity_counts=entity_counts,
        )

    except Exception as e:
        logger.warning(f"PII scan failed — returning original text: {e}")
        return PIIScanResult(
            original_text=text,
            redacted_text=text,
            findings=[],
            pii_found=False,
            entity_counts={},
        )


def redact(
    text: str,
    score_threshold: float = 0.6,
    placeholder_format: str = "[{entity_type}]",
) -> str:
    """
    Redact PII from text. Replaces detected entities with type labels.

    Examples:
      "John Doe signed"               → "[PERSON] signed"
      "Email john@acme.com for info"  → "Email [EMAIL_ADDRESS] for info"
      "Call +1-555-123-4567"          → "Call [PHONE_NUMBER]"
      "CC: 4111 1111 1111 1111"       → "CC: [CREDIT_CARD]"

    Args:
        text:               text to redact
        score_threshold:    minimum confidence to redact
        placeholder_format: format string for replacement label

    Returns:
        redacted text string
    """
    if not text or not settings.enable_pii_redaction:
        return text

    try:
        analyzer = _get_analyzer()
        results  = analyzer.analyze(
            text=text,
            entities=settings.pii_entities,
            language="en",
            score_threshold=score_threshold,
        )

        if not results:
            return text

        redacted = _redact_text(text, results, placeholder_format)

        redacted_count = len(results)
        logger.debug(f"Redacted {redacted_count} PII entities from text")
        return redacted

    except Exception as e:
        logger.warning(f"PII redaction failed — returning original text: {e}")
        return text


def _redact_text(
    text: str,
    results: list,
    placeholder_format: str = "[{entity_type}]",
) -> str:
    """Replace detected PII spans with placeholder labels."""
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    anonymizer = _get_anonymizer()

    operators = {
        entity: OperatorConfig(
            "replace",
            {"new_value": placeholder_format.format(entity_type=entity)},
        )
        for entity in settings.pii_entities
    }

    result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return result.text


def redact_chunk_list(chunks: list) -> list:
    """
    Redact PII from a list of LangChain Document chunks.
    Used during ingestion before chunks are embedded.

    Args:
        chunks: list of LangChain Document objects

    Returns:
        same list with page_content redacted in-place
    """
    if not settings.enable_pii_redaction:
        return chunks

    redacted_count = 0
    for chunk in chunks:
        original = chunk.page_content
        chunk.page_content = redact(original)
        if chunk.page_content != original:
            redacted_count += 1

    if redacted_count:
        logger.info(f"PII redacted in {redacted_count}/{len(chunks)} chunks")

    return chunks
