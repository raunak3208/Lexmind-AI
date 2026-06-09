"""
evaluation/metrics.py

Standalone metric implementations that work WITHOUT calling RAGAS.
These run locally using sentence-transformers similarity — no API cost.

Metrics:
  context_recall      did retrieval find chunks containing the answer?
  context_precision   how much of retrieved context was actually useful?
  answer_relevancy    does the answer address the question?
  faithfulness_score  is the answer grounded in the retrieved context?

These complement RAGAS — useful when you want fast local evaluation
without any LLM judge calls.

Location: evaluation/metrics.py
"""

import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_similarity_model():
    """Load sentence-transformers model for semantic similarity (cached)."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading similarity model for metrics: all-MiniLM-L6-v2")
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.
    Uses sentence-transformers if available, falls back to token overlap.
    """
    import numpy as np
    try:
        model = _get_similarity_model()
        embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
    except Exception:
        # Fallback: Jaccard token overlap when model not available
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return round(len(intersection) / len(union), 4)


def context_recall(
    ground_truth: str,
    contexts: list[str],
    threshold: float = 0.5,
) -> float:
    """
    Measure how much of the ground truth is covered by retrieved contexts.

    Score 1.0 = ground truth is well represented in retrieved chunks.
    Score 0.0 = ground truth not found in any retrieved chunk.

    Method: semantic similarity between ground_truth and each context chunk.
    Returns max similarity across all chunks (best-match recall).

    Args:
        ground_truth: correct answer string
        contexts:     list of retrieved chunk texts
        threshold:    similarity threshold to count as a hit

    Returns:
        float 0.0 to 1.0
    """
    if not contexts or not ground_truth:
        return 0.0

    scores = [_cosine_similarity(ground_truth, ctx) for ctx in contexts]
    best_score = max(scores)

    logger.debug(f"context_recall: best={best_score:.3f} threshold={threshold}")
    return round(best_score, 4)


def context_precision(
    question: str,
    contexts: list[str],
    threshold: float = 0.4,
) -> float:
    """
    Measure what fraction of retrieved contexts are actually relevant to the question.

    Score 1.0 = every retrieved chunk is relevant.
    Score 0.0 = no retrieved chunk is relevant.

    Args:
        question:  the user question
        contexts:  list of retrieved chunk texts
        threshold: similarity threshold to count a chunk as relevant

    Returns:
        float 0.0 to 1.0
    """
    if not contexts or not question:
        return 0.0

    relevant_count = sum(
        1 for ctx in contexts
        if _cosine_similarity(question, ctx) >= threshold
    )

    score = relevant_count / len(contexts)
    logger.debug(
        f"context_precision: {relevant_count}/{len(contexts)} relevant  score={score:.3f}"
    )
    return round(score, 4)


def answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """
    Measure how relevant the generated answer is to the question.

    Score 1.0 = answer directly addresses the question.
    Score 0.0 = answer is completely off-topic.

    Args:
        question: the user question
        answer:   the RAG-generated answer

    Returns:
        float 0.0 to 1.0
    """
    if not question or not answer:
        return 0.0

    score = _cosine_similarity(question, answer)
    logger.debug(f"answer_relevancy: score={score:.3f}")
    return round(score, 4)


def faithfulness_score(
    answer: str,
    contexts: list[str],
) -> float:
    """
    Measure how well the answer is grounded in the retrieved contexts.
    A faithful answer should only contain information from the contexts.

    Score 1.0 = answer is fully supported by retrieved context.
    Score 0.0 = answer appears to hallucinate — not in context.

    Method: semantic similarity between answer and best matching context chunk.

    Args:
        answer:   the RAG-generated answer
        contexts: list of retrieved chunk texts

    Returns:
        float 0.0 to 1.0
    """
    if not answer or not contexts:
        return 0.0

    scores = [_cosine_similarity(answer, ctx) for ctx in contexts]
    best_score = max(scores)

    logger.debug(f"faithfulness: best_context_match={best_score:.3f}")
    return round(best_score, 4)


def compute_all_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """
    Compute all 4 metrics for a single Q&A sample.

    Returns:
        dict with keys:
          context_recall, context_precision, answer_relevancy,
          faithfulness, composite_score
    """
    cr  = context_recall(ground_truth, contexts)
    cp  = context_precision(question, contexts)
    ar  = answer_relevancy(question, answer)
    fth = faithfulness_score(answer, contexts)

    # Composite = weighted average
    composite = round((cr * 0.3 + cp * 0.2 + ar * 0.25 + fth * 0.25), 4)

    return {
        "context_recall":    cr,
        "context_precision": cp,
        "answer_relevancy":  ar,
        "faithfulness":      fth,
        "composite_score":   composite,
    }