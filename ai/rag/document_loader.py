"""
ai/rag/document_loader.py

Loads PDF / DOCX contracts → splits into overlapping chunks.
Returns a list of LangChain Document objects ready for embedding.

Supported formats:  .pdf  |  .docx  |  .txt
"""

import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from ai.config import settings

logger = logging.getLogger(__name__)


# ── Supported extensions ──────────────────────────────────────────────────────
LOADERS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}


def load_document(file_path: str) -> List[Document]:
    """
    Load a contract file from disk and return raw LangChain Documents
    (one per page for PDF, one for DOCX/TXT).

    Args:
        file_path: absolute path to the uploaded file

    Returns:
        list of LangChain Document objects
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(LOADERS.keys())}"
        )

    loader_cls = LOADERS[ext]
    loader = loader_cls(str(path))

    logger.info(f"Loading document: {path.name}  (type={ext})")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} page(s) from {path.name}")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents into overlapping text chunks for RAG.

    Legal contracts are verbose → we use RecursiveCharacterTextSplitter which
    respects paragraph / sentence / word boundaries in that order.

    Args:
        documents:     raw LangChain Documents from load_document()
        chunk_size:    max chars per chunk (default from config)
        chunk_overlap: overlap between chunks (default from config)

    Returns:
        list of chunked LangChain Document objects
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Legal docs use these as natural break points
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Chunked into {len(chunks)} pieces "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def load_and_chunk(file_path: str) -> List[Document]:
    """
    Convenience wrapper: load + chunk in one call.

    Args:
        file_path: absolute path to the uploaded contract file

    Returns:
        list of chunked LangChain Documents ready for embedding
    """
    documents = load_document(file_path)
    chunks    = chunk_documents(documents)
    return chunks


def enrich_metadata(
    chunks: List[Document],
    document_id: str,
    filename: str,
    uploaded_by: str = "unknown",
) -> List[Document]:
    """
    Stamp each chunk with document-level metadata so we can filter by
    document_id later in the vector store.

    Args:
        chunks:        output of load_and_chunk()
        document_id:   unique doc ID (assigned by Node backend / MongoDB)
        filename:      original filename shown to the user
        uploaded_by:   user ID from Node's auth layer

    Returns:
        same chunks with enriched .metadata dict
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "document_id":  document_id,
                "filename":     filename,
                "uploaded_by":  uploaded_by,
                "chunk_index":  i,
                "total_chunks": len(chunks),
            }
        )
    return chunks
