"""
ai/config.py
Central config — reads from ai/.env  (copy .env.example → .env and fill in keys)
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # ── Mistral (FREE tier) ────────────────────────────────────────────────
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")

    # LLM model  — mistral-small-latest is FREE on Mistral's free tier
    mistral_llm_model: str = "mistral-small-latest"

    # Embedding model — mistral-embed is FREE
    mistral_embed_model: str = "mistral-embed"

    # BGE Embeddings — runs locally, no API cost, better than mistral-embed
    use_bge_embeddings: bool = True
    bge_model_name: str = "BAAI/bge-small-en-v1.5"


    # Hybrid Search settings
    use_hybrid_search: bool = True
    bm25_index_dir: str = str(BASE_DIR.parent / "data" / "bm25_index")
    hybrid_dense_weight: float = 0.5
    hybrid_sparse_weight: float = 0.5
    hybrid_rrf_k: int = 60

    # Reranker — cross-encoder, runs locally, no API cost
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3



    # Semantic Cache (Redis)
    enable_semantic_cache: bool = True
    redis_url: str = "redis://localhost:6379"
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 86400
    cache_max_size: int = 10000
    cache_key_prefix: str = "lexmind:cache"
    cache_embed_dim: int = 384

    # Guardrails + PII settings
    enable_guardrails: bool = True
    enable_pii_redaction: bool = True
    pii_redaction_mode: str = "replace"   # replace | mask | hash
    max_query_length: int = 2000
    min_query_length: int = 3
    prompt_injection_threshold: float = 0.85
    pii_entities: list = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
        "LOCATION", "DATE_TIME", "NRP",
        "MEDICAL_LICENSE", "URL",
    ]


    # Knowledge Graph (GraphRAG)
    enable_knowledge_graph: bool = True
    graph_persist_dir: str = str(BASE_DIR.parent / "data" / "knowledge_graphs")
    knowledge_graph_dir: str = str(BASE_DIR.parent / "data" / "knowledge_graphs")
    graph_max_entities_per_chunk: int = 15
    graph_entity_types: list = [
        "PERSON", "ORGANIZATION", "CLAUSE", "OBLIGATION",
        "PAYMENT", "DATE", "AMOUNT", "JURISDICTION",
        "PENALTY", "CONDITION", "RIGHT", "RESTRICTION"
    ]

    # ── Vector DB (local Chroma — 100% free) ──────────────────────────────
    chroma_persist_dir: str = str(BASE_DIR.parent / "data" / "vector_db")

    # ── File Storage ──────────────────────────────────────────────────────
    upload_dir: str = str(BASE_DIR.parent / "data" / "uploads")

    # ── FastAPI ───────────────────────────────────────────────────────────
    ai_host: str = "0.0.0.0"
    ai_port: int = 8000

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Retriever ─────────────────────────────────────────────────────────
    retriever_k: int = 5          # top-k chunks to retrieve

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import this everywhere
settings = Settings()
