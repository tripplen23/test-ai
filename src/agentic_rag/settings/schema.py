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
    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    dataset: DatasetConfig = DatasetConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    ingestion_class: Optional[str] = None
    agent_controller_class: Optional[str] = None
    evaluator_class: Optional[str] = None


_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
