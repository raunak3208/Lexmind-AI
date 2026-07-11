"""
ai/api/routes/ingest.py

POST /ingest
Called by Node backend after a file is saved to disk.
Loads the file, chunks it, embeds with Mistral, stores in Chroma.
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.rag.document_loader import load_and_chunk, enrich_metadata
from ai.rag.vector_store    import add_chunks
from ai.guardrails.guardrail_service import protect_chunks, protect_contract

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    file_path:   str    # absolute path on disk — set by Node after saving upload
    document_id: str    # MongoDB _id from Node backend
    filename:    str    # original filename for display
    uploaded_by: str = "unknown"


class IngestResponse(BaseModel):
    document_id:  str
    filename:     str
    total_chunks: int
    vector_ids:   list[str]
    status:       str = "ingested"


@router.post("", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    """
    Ingest a contract file into the vector store.

    Steps:
    1. Load file (PDF / DOCX / TXT) from disk
    2. Split into overlapping chunks
    3. Enrich chunks with document metadata
    4. Embed with Mistral + store in Chroma
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    logger.info(f"Ingesting document_id={req.document_id}  file={req.filename}")

    try:
        # 1+2. Load & chunk
        chunks = load_and_chunk(req.file_path)

        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

        # 3. Enrich metadata
        chunks = enrich_metadata(
            chunks,
            document_id=req.document_id,
            filename=req.filename,
            uploaded_by=req.uploaded_by,
        )

        # 3b. Redact PII from chunks before embedding
        # Ensures PII never enters the vector store
        chunks = protect_chunks(chunks)

        # 4. Embed & store
        vector_ids = add_chunks(chunks)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingestion failed for {req.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    logger.info(f"Ingested {len(vector_ids)} vectors for {req.document_id}")
    return IngestResponse(
        document_id=req.document_id,
        filename=req.filename,
        total_chunks=len(chunks),
        vector_ids=vector_ids,
    )
