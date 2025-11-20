# RAG Retrieval Pipeline Challenge

You'll receive a messy WordPress QA dataset and transform it into a functional retrieval system. We provide the scaffolding (CLI commands, settings, optional Docker infrastructure). The implementation is entirely yours.

## Core Requirements

The task centers on two components. First, build an ingestion pipeline that cleans the raw data, chunks it appropriately, and persists it to the provided Postgres + pgvector instance (see `docker-compose.yml`). If you prefer an alternative to pgvector, document your reasoning. Second, design and implement the retrieval system itself: embedding strategy, indexing approach, and query handling. We intentionally provide no defaults here.

The core work should take 2-3 hours. If you finish early or want to demonstrate additional capabilities, consider adding a reranker or implementing a multi-turn conversational agent. These are strictly optional.

## Repository Structure
```
.
├── Makefile                  
├── docker-compose.yml        
├── pyproject.toml            
├── src/agentic_rag/          
└── scripts/download_dataset.py
```

Reorganize as needed, but preserve the CLI commands for reproducibility.

## Setup and Execution
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make compose-up        
make data              
make ingest            
make agent             
```

Stop the pgvector service with `make compose-down`. If the dataset requires authentication, export `HF_TOKEN` before downloading.

## Configuration

Specify your implementations via environment variables or `.env`:
```bash
export AGENTIC_RAG_INGESTION_CLASS="my_pkg.ingestion.Pipeline"
export AGENTIC_RAG_AGENT_CONTROLLER_CLASS="my_pkg.agent.Controller"
```

## Submission

Provide a repository link with clear execution instructions. Include a brief architectural overview that addresses your design decisions: embedding model selection, chunking strategy, pgvector schema and indexing approach, known limitations, and potential improvements. If you deviated from pgvector, explain the alternative and its trade-offs.

Additional artifacts (tests, documentation, tooling) are at your discretion.
