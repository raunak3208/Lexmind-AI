from types import SimpleNamespace

import pytest

from ai.config import settings
from ai.guardrails import pii_detector
from ai.guardrails.output_guard import MAX_ANSWER_LENGTH, guard_output


@pytest.fixture(autouse=True)
def guardrail_settings(monkeypatch):
    monkeypatch.setattr(settings, "enable_guardrails", True)
    monkeypatch.setattr(settings, "enable_pii_redaction", False)


def test_guard_output_passes_through_when_guardrails_are_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_guardrails", False)

    result = guard_output("short")

    assert result.is_safe is True
    assert result.final_text == "short"


@pytest.mark.parametrize("answer", ["", "Too short"])
def test_guard_output_replaces_empty_or_near_empty_answers(answer):
    result = guard_output(answer)

    assert result.is_safe is False
    assert result.final_text.startswith("The system could not generate")
    assert result.warnings == ["Empty response from LLM"]


def test_guard_output_returns_safe_answers_unchanged():
    answer = "The payment clause requires settlement within thirty days."

    result = guard_output(answer)

    assert result.is_safe is True
    assert result.final_text == answer
    assert result.warnings == []
    assert result.pii_leaked is False


def test_guard_output_redacts_detected_pii(monkeypatch):
    monkeypatch.setattr(settings, "enable_pii_redaction", True)
    monkeypatch.setattr(
        pii_detector,
        "scan",
        lambda answer: SimpleNamespace(
            pii_found=True,
            redacted_text="Contact [EMAIL_ADDRESS] about the agreement.",
            entity_counts={"EMAIL_ADDRESS": 1},
        ),
    )

    result = guard_output("Contact counsel@example.com about the agreement.")

    assert result.is_safe is True
    assert result.final_text == "Contact [EMAIL_ADDRESS] about the agreement."
    assert result.pii_leaked is True
    assert result.pii_redacted_output is True
    assert "EMAIL_ADDRESS" in result.warnings[0]


def test_guard_output_flags_hallucination_signals():
    result = guard_output(
        "As of my knowledge cutoff, this clause may have a different meaning."
    )

    assert result.is_safe is True
    assert result.hallucination_risk is True
    assert len(result.warnings) == 1


def test_guard_output_identifies_refusals_without_blocking():
    answer = "I cannot help answer this request using the supplied contract."

    result = guard_output(answer)

    assert result.is_safe is True
    assert result.was_refusal is True
    assert result.final_text == answer


def test_guard_output_truncates_excessively_long_answers():
    result = guard_output("x" * (MAX_ANSWER_LENGTH + 10))

    assert result.is_safe is True
    assert result.final_text.endswith("[Response truncated]")
    assert result.warnings == [
        "Response was truncated to maximum allowed length."
    ]
