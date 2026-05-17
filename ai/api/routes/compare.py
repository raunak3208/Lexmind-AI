"""
ai/api/routes/compare.py

POST /compare
Compare two ingested contracts side-by-side using vector similarity.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.tools.comparison_tool import compare_contracts, ComparisonResult

router = APIRouter()
logger = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    document_id_a: str
    filename_a:    str
    document_id_b: str
    filename_b:    str


@router.post("", response_model=ComparisonResult)
async def compare_two_contracts(req: CompareRequest):
    """
    Compare two contracts clause-by-clause.
    Both documents must already be ingested into the vector store.

    Returns a side-by-side comparison for all standard clause types
    with a summary of what's missing in each.
    """
    if req.document_id_a == req.document_id_b:
        raise HTTPException(
            status_code=422,
            detail="document_id_a and document_id_b must be different documents.",
        )

    logger.info(
        f"Comparing {req.document_id_a} ({req.filename_a}) "
        f"vs {req.document_id_b} ({req.filename_b})"
    )

    try:
        result = await compare_contracts(
            document_id_a=req.document_id_a,
            filename_a=req.filename_a,
            document_id_b=req.document_id_b,
            filename_b=req.filename_b,
        )
    except Exception as e:
        logger.exception(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

    return result