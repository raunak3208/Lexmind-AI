"""
ai/api/routes/evaluate.py

POST /evaluate        run evaluation and return scores
POST /evaluate/compare  compare multiple retrieval strategies

Location: ai/api/routes/evaluate.py
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class EvalRequest(BaseModel):
    document_id: Optional[str] = None
    strategy:    str = "mmr"
    mode:        str = "local"
    max_samples: Optional[int] = None


class CompareRequest(BaseModel):
    document_id: Optional[str] = None
    strategies:  list[str] = ["mmr", "reranked", "similarity"]
    mode:        str = "local"
    max_samples: Optional[int] = None


@router.post("")
async def run_evaluation(req: EvalRequest):
    """
    Run RAG evaluation on ingested documents.

    mode=local  — uses sentence-transformers, no API cost, runs in seconds
    mode=ragas  — uses Mistral as judge, more accurate, costs API tokens
    """
    if req.mode not in ("local", "ragas"):
        raise HTTPException(status_code=422, detail="mode must be 'local' or 'ragas'")

    logger.info(
        f"Evaluation requested: mode={req.mode}  "
        f"strategy={req.strategy}  doc={req.document_id or 'ALL'}"
    )

    try:
        if req.mode == "ragas":
            from evaluation.ragas_evaluator import run_ragas_evaluation
            result = await run_ragas_evaluation(
                document_id=req.document_id,
                strategy=req.strategy,
                max_samples=req.max_samples,
            )
        else:
            from evaluation.ragas_evaluator import run_local_evaluation
            result = await run_local_evaluation(
                document_id=req.document_id,
                strategy=req.strategy,
                max_samples=req.max_samples,
            )

        return {"status": "completed", "result": result}

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_strategies_endpoint(req: CompareRequest):
    """
    Compare multiple retrieval strategies side by side.
    Returns scores per strategy and the winner.
    """
    if len(req.strategies) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 strategies to compare")

    logger.info(f"Strategy comparison: {req.strategies}  mode={req.mode}")

    try:
        from evaluation.ragas_evaluator import compare_strategies
        result = await compare_strategies(
            strategies=req.strategies,
            document_id=req.document_id,
            mode=req.mode,
            max_samples=req.max_samples,
        )
        return {"status": "completed", "result": result}

    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))