"""
ai/api/main.py

FastAPI entry point for the LexMind AI service.
Node.js backend calls this service via HTTP.

Routes:
  POST /ingest    — upload & embed a contract
  POST /analyze   — run full 4-agent pipeline
  POST /search    — semantic clause search
  POST /chat      — per-document conversation
  POST /compare   — side-by-side contract comparison
  GET  /health    — health check + vector store stats
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.api.routes import ingest, analyze, search, chat, compare

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logger.info("=== LexMind AI Service starting ===")
    # Ensure data directories exist
    from ai.config import settings
    os.makedirs(settings.upload_dir,       exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    logger.info(f"Upload dir  : {settings.upload_dir}")
    logger.info(f"Vector DB   : {settings.chroma_persist_dir}")
    logger.info(f"LLM model   : {settings.mistral_llm_model}")
    logger.info(f"Embed model : {settings.mistral_embed_model}")
    yield
    logger.info("=== LexMind AI Service shutting down ===")


# App 
app = FastAPI(
    title="LexMind AI Service",
    description="Multi-agent RAG pipeline for legal contract analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Node backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers 
app.include_router(ingest.router,  prefix="/ingest",  tags=["Ingestion"])
app.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
app.include_router(search.router,  prefix="/search",  tags=["Search"])
app.include_router(chat.router,    prefix="/chat",    tags=["Chat"])
app.include_router(compare.router, prefix="/compare", tags=["Comparison"])


@app.get("/health", tags=["Health"])
async def health():
    """Health check — returns service status and vector store stats."""
    from ai.rag.vector_store import get_collection_stats
    stats = get_collection_stats()
    return {"status": "ok", "vector_store": stats}