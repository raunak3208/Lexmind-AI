"""
ai/api/routes/analyze.py

POST /analyze
Triggers the full multi-agent pipeline on an already-ingested document.
Returns FullAnalysisResult (clauses + risk report + summary).
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.rag.document_loader   import load_document
from ai.agents.orchestrator   import run_full_pipeline
from ai.schemas.document_schema import FullAnalysisResult

router = APIRouter()
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    file_path:   str   # path on disk (same as used in /ingest)
    document_id: str
    filename:    str


@router.post("", response_model=FullAnalysisResult)
async def analyze_document(req: AnalyzeRequest):
    """
    Run the 4-agent pipeline:
      Extractor → Classifier → Risk Agent → Summarizer

    The document must already be ingested (vectors in Chroma).
    This endpoint re-reads the raw text for the agents.
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    logger.info(f"Analysis requested for document_id={req.document_id}")

    try:
        # Load raw text (not chunked — agents need the full contract)
        docs = load_document(req.file_path)
        contract_text = "\n\n".join(d.page_content for d in docs)

        if not contract_text.strip():
            raise HTTPException(status_code=422, detail="No text extracted from file.")

        # Run full pipeline
        result = await run_full_pipeline(
            contract_text=contract_text,
            document_id=req.document_id,
            filename=req.filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis pipeline failed for {req.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    return result
