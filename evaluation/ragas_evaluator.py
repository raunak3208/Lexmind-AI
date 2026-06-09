"""
evaluation/ragas_evaluator.py

Main RAGAS evaluation runner for LexMind.

Runs two evaluation modes:

  Mode 1 — RAGAS full evaluation (uses Mistral as judge LLM):
    Metrics: faithfulness, answer_relevancy, context_recall, context_precision
    Requires: MISTRAL_API_KEY, ingested documents in Chroma

  Mode 2 — Local lightweight evaluation (no LLM judge, uses sentence-transformers):
    Uses evaluation/metrics.py
    Requires: only sentence-transformers (already installed)

Usage:
  # Full RAGAS evaluation
  python -m evaluation.ragas_evaluator --document_id <id> --mode ragas

  # Local evaluation (no extra API calls)
  python -m evaluation.ragas_evaluator --document_id <id> --mode local

  # Compare two retrieval strategies
  python -m evaluation.ragas_evaluator --document_id <id> --compare mmr reranked

Location: evaluation/ragas_evaluator.py
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


async def run_ragas_evaluation(
    document_id: Optional[str] = None,
    strategy: str = "mmr",
    max_samples: int = None,
    test_cases_path: str = None,
) -> dict:
    """
    Run full RAGAS evaluation using Mistral as judge LLM.

    RAGAS metrics:
      faithfulness        — is answer grounded in context (no hallucination)
      answer_relevancy    — does answer address the question
      context_recall      — did retrieval find the right chunks
      context_precision   — how precise is the retrieved context

    Args:
        document_id:      MongoDB document ID to evaluate against
        strategy:         retrieval strategy to test
        max_samples:      limit test cases (None = all 10)
        test_cases_path:  custom test cases JSON path

    Returns:
        dict with scores and per-sample results
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from langchain_mistralai import ChatMistralAI
    from langchain_mistralai import MistralAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ai.config import settings
    from evaluation.dataset import build_eval_dataset

    logger.info(f"RAGAS evaluation: strategy={strategy}  doc={document_id or 'ALL'}")

    dataset = await build_eval_dataset(
        document_id=document_id,
        strategy=strategy,
        max_samples=max_samples,
        test_cases_path=test_cases_path,
    )

    logger.info(f"Dataset ready: {len(dataset)} samples — running RAGAS...")

    # Use Mistral as judge (free tier)
    judge_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=settings.mistral_llm_model,
            mistral_api_key=settings.mistral_api_key,
            temperature=0,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model=settings.mistral_embed_model,
            mistral_api_key=settings.mistral_api_key,
        )
    )

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )

    scores = {
        "faithfulness":        round(float(result["faithfulness"]), 4),
        "answer_relevancy":    round(float(result["answer_relevancy"]), 4),
        "context_recall":      round(float(result["context_recall"]), 4),
        "context_precision":   round(float(result["context_precision"]), 4),
        "composite":           round(
            (result["faithfulness"] + result["answer_relevancy"] +
             result["context_recall"] + result["context_precision"]) / 4, 4
        ),
    }

    return {
        "mode":        "ragas",
        "strategy":    strategy,
        "document_id": document_id,
        "n_samples":   len(dataset),
        "scores":      scores,
        "timestamp":   datetime.utcnow().isoformat(),
        "dataframe":   result.to_pandas().to_dict(orient="records"),
    }


async def run_local_evaluation(
    document_id: Optional[str] = None,
    strategy: str = "mmr",
    max_samples: int = None,
    test_cases_path: str = None,
) -> dict:
    """
    Run local evaluation using sentence-transformers (no LLM judge, no API cost).
    Uses evaluation/metrics.py compute_all_metrics().

    Args:
        document_id:      MongoDB document ID to evaluate against
        strategy:         retrieval strategy to test
        max_samples:      limit test cases
        test_cases_path:  custom test cases JSON path

    Returns:
        dict with per-sample and aggregate scores
    """
    from evaluation.dataset import build_eval_dataset
    from evaluation.metrics import compute_all_metrics

    logger.info(f"Local evaluation: strategy={strategy}  doc={document_id or 'ALL'}")

    dataset = await build_eval_dataset(
        document_id=document_id,
        strategy=strategy,
        max_samples=max_samples,
        test_cases_path=test_cases_path,
    )

    per_sample = []
    aggregate = {
        "context_recall":    0.0,
        "context_precision": 0.0,
        "answer_relevancy":  0.0,
        "faithfulness":      0.0,
        "composite_score":   0.0,
    }

    for i, row in enumerate(dataset):
        m = compute_all_metrics(
            question=row["question"],
            answer=row["answer"],
            contexts=row["contexts"],
            ground_truth=row["ground_truth"],
        )

        logger.info(
            f"  [{i+1}/{len(dataset)}] composite={m['composite_score']:.3f}  "
            f"recall={m['context_recall']:.3f}  "
            f"precision={m['context_precision']:.3f}  "
            f"relevancy={m['answer_relevancy']:.3f}  "
            f"faithful={m['faithfulness']:.3f}"
        )

        per_sample.append({
            "question":    row["question"],
            "answer":      row["answer"],
            "ground_truth": row["ground_truth"],
            "metrics":     m,
        })

        for key in aggregate:
            aggregate[key] += m[key]

    n = len(dataset)
    for key in aggregate:
        aggregate[key] = round(aggregate[key] / n, 4)

    return {
        "mode":        "local",
        "strategy":    strategy,
        "document_id": document_id,
        "n_samples":   n,
        "scores":      aggregate,
        "per_sample":  per_sample,
        "timestamp":   datetime.utcnow().isoformat(),
    }


async def compare_strategies(
    strategies: list[str],
    document_id: Optional[str] = None,
    mode: str = "local",
    max_samples: int = None,
) -> dict:
    """
    Compare multiple retrieval strategies side by side.

    Args:
        strategies:   list of strategy names e.g. ["mmr", "reranked", "similarity"]
        document_id:  document to evaluate against
        mode:         "local" or "ragas"
        max_samples:  limit test cases per strategy

    Returns:
        dict with results per strategy and winner
    """
    results = {}

    for strategy in strategies:
        logger.info(f"Evaluating strategy: {strategy}")
        if mode == "ragas":
            result = await run_ragas_evaluation(
                document_id=document_id,
                strategy=strategy,
                max_samples=max_samples,
            )
        else:
            result = await run_local_evaluation(
                document_id=document_id,
                strategy=strategy,
                max_samples=max_samples,
            )
        results[strategy] = result

    # Determine winner by composite score
    winner = max(
        results.keys(),
        key=lambda s: results[s]["scores"].get("composite_score", results[s]["scores"].get("composite", 0))
    )

    logger.info(f"Winner: {winner}")

    return {
        "comparison": results,
        "winner":     winner,
        "timestamp":  datetime.utcnow().isoformat(),
    }


def save_result(result: dict, filename: str = None) -> str:
    """Save evaluation result to JSON file."""
    if not filename:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mode = result.get("mode", result.get("comparison", {}) and "comparison")
        strategy = result.get("strategy", "multi")
        filename = f"eval_{mode}_{strategy}_{ts}.json"

    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Result saved: {path}")
    return str(path)


async def main():
    parser = argparse.ArgumentParser(description="LexMind RAG Evaluator")
    parser.add_argument("--document_id", type=str, default=None,
                        help="MongoDB document ID to evaluate against")
    parser.add_argument("--mode", type=str, default="local",
                        choices=["local", "ragas"],
                        help="Evaluation mode: local (no API) or ragas (uses Mistral)")
    parser.add_argument("--strategy", type=str, default="mmr",
                        help="Retrieval strategy to evaluate")
    parser.add_argument("--compare", type=str, nargs="+",
                        help="Compare multiple strategies e.g. --compare mmr reranked similarity")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test cases")
    parser.add_argument("--test_cases", type=str, default=None,
                        help="Path to custom test cases JSON")
    parser.add_argument("--save", action="store_true",
                        help="Save results to evaluation/results/")

    args = parser.parse_args()

    if args.compare:
        result = await compare_strategies(
            strategies=args.compare,
            document_id=args.document_id,
            mode=args.mode,
            max_samples=args.max_samples,
        )
        print(f"\nWinner: {result['winner']}")
        for strategy, r in result["comparison"].items():
            scores = r["scores"]
            print(f"\n{strategy}:")
            for k, v in scores.items():
                print(f"  {k}: {v}")
    else:
        if args.mode == "ragas":
            result = await run_ragas_evaluation(
                document_id=args.document_id,
                strategy=args.strategy,
                max_samples=args.max_samples,
                test_cases_path=args.test_cases,
            )
        else:
            result = await run_local_evaluation(
                document_id=args.document_id,
                strategy=args.strategy,
                max_samples=args.max_samples,
                test_cases_path=args.test_cases,
            )

        print(f"\nEvaluation Results ({args.mode} mode, strategy={args.strategy})")
        print("-" * 50)
        for k, v in result["scores"].items():
            bar = "#" * int(v * 20)
            print(f"  {k:<22} {v:.4f}  [{bar:<20}]")

    if args.save:
        path = save_result(result)
        print(f"\nSaved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())