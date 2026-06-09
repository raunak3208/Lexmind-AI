"""
evaluation/dataset.py

Builds a RAGAS-compatible evaluation dataset.

RAGAS needs 4 things per sample:
  question      - what the user asked
  answer        - what the RAG system returned
  contexts      - list of chunks that were retrieved
  ground_truth  - the correct answer (from legal_qa.json)

This module:
  1. Loads Q&A pairs from test_cases/legal_qa.json
  2. For each question, runs the RAG pipeline to get answer + contexts
  3. Returns a Dataset object RAGAS can evaluate

Location: evaluation/dataset.py
"""

import json
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY", ""))

logger = logging.getLogger(__name__)

TEST_CASES_PATH = Path(__file__).parent / "test_cases" / "legal_qa.json"


def load_test_cases(path: str = None) -> list[dict]:
    """Load Q&A test cases from JSON file."""
    p = Path(path) if path else TEST_CASES_PATH
    with open(p, "r") as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} test cases from {p}")
    return cases


async def _get_answer_and_contexts(
    question: str,
    document_id: Optional[str],
    strategy: str = "mmr",
) -> tuple[str, list[str]]:
    """
    Run RAG pipeline for one question.
    Returns (answer, list_of_context_strings).
    """
    from ai.rag.retriever import get_retriever
    from ai.rag.rag_chain import ask, build_rag_chain
    from ai.rag.vector_store import similarity_search

    # Get contexts separately so we can pass them to RAGAS
    retriever = get_retriever(strategy=strategy, document_id=document_id)
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    # Get answer from RAG chain
    result = await ask(
        question=question,
        document_id=document_id,
        strategy=strategy,
    )
    answer = result["answer"]

    return answer, contexts


async def build_eval_dataset(
    document_id: Optional[str] = None,
    test_cases_path: str = None,
    strategy: str = "mmr",
    max_samples: int = None,
) -> Dataset:
    """
    Build a RAGAS evaluation Dataset by running the RAG pipeline
    on every test case question.

    Args:
        document_id:      restrict retrieval to one document (recommended)
                          None = search all documents
        test_cases_path:  path to JSON file with Q&A pairs
        strategy:         retrieval strategy to evaluate
        max_samples:      limit number of test cases (None = all)

    Returns:
        HuggingFace Dataset with columns:
        question, answer, contexts, ground_truth
    """
    test_cases = load_test_cases(test_cases_path)

    if max_samples:
        test_cases = test_cases[:max_samples]

    questions    = []
    answers      = []
    contexts_list = []
    ground_truths = []

    logger.info(
        f"Building eval dataset: {len(test_cases)} questions  "
        f"strategy={strategy}  doc={document_id or 'ALL'}"
    )

    for i, case in enumerate(test_cases):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        logger.info(f"  [{i+1}/{len(test_cases)}] {question[:60]}")

        try:
            answer, contexts = await _get_answer_and_contexts(
                question=question,
                document_id=document_id,
                strategy=strategy,
            )
        except Exception as e:
            logger.warning(f"  Failed: {e} — using empty answer")
            answer   = ""
            contexts = []

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    logger.info(f"Dataset built: {len(dataset)} samples")
    return dataset