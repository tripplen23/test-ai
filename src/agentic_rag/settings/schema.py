from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreConfig(BaseModel):
    implementation: Optional[str] = Field(
        default=None,
        description="Name of the vector store backend (e.g., pgvector, chroma).",
    )
    collection: str = Field(default="wordpress", description="Vector collection name.")
    embedding_model: Optional[str] = Field(default=None)
    cross_encoder_model: Optional[str] = Field(default=None)


class TelemetryConfig(BaseModel):
    enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)


class DatasetConfig(BaseModel):
    name: str = Field(default="mteb/cqadupstack-wordpress")
    corpus_filename: str = Field(default="corpus.jsonl")
    queries_filename: str = Field(default="queries.jsonl")
    qrels_filename: str = Field(default="qrels.jsonl")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,  # Allow reading from alias names
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    dataset: DatasetConfig = DatasetConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    ingestion_class: Optional[str] = Field(
        default="agentic_rag.data.ingestion_pipeline.IngestionPipeline",
        description="Full path to ingestion pipeline class"
    )
    agent_controller_class: Optional[str] = None
    evaluator_class: Optional[str] = None
    
    # Hugging Face configuration
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for authenticated datasets",
        alias="HF_TOKEN", 
    )
    
    # OpenAI configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings and models",
        alias="OPENAI_API_KEY",
    )
    
    # Tavily configuration
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search",
        alias="TAVILY_API_KEY",
    )
    
    # Database configuration
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_user: str = Field(default="rag", description="PostgreSQL user")
    db_password: str = Field(default="rag", description="PostgreSQL password")
    db_name: str = Field(default="rag", description="PostgreSQL database name")
    
    # Embedding configuration
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model name")
    embedding_dimension: Optional[int] = Field(default=1536, description="OpenAI embedding dimension (text-embedding-3-small = 1536)")
    
    # Chunking configuration
    chunk_size: int = Field(default=1000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Chunk overlap in characters")
    
    # Retrieval configuration
    retrieval_top_k: int = Field(default=5, description="Number of documents to retrieve")
    retrieval_score_threshold: float = Field(default=0.5, description="Minimum similarity score (0.0-1.0)")
    
    # Reranker configuration (optional)
    reranker_enabled: bool = Field(default=False, description="Enable cross-encoder reranking")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model name")
    reranker_top_k: int = Field(default=3, description="Final number of results after reranking")
    
    # Agent configuration
    agent_max_history: int = Field(default=5, description="Maximum conversation history length")


_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings