"""
ai/config.py
Central config
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Mistral API
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    mistral_llm_model: str = "mistral-small-latest"
    mistral_embed_model: str = "mistral-embed"

    # Local storage
    chroma_persist_dir: str = str(BASE_DIR.parent / "data" / "vector_db")
    upload_dir: str = str(BASE_DIR.parent / "data" / "uploads")

    # FastAPI
    ai_host: str = "0.0.0.0"
    ai_port: int = 8000

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 5

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()