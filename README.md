# Agentic RAG Challenge

You receive a messy WordPress QA dataset (via `make data`). Turn it into a useful knowledge base and expose an agent that can reason over multi-turn requests. The scaffold only wires up CLI entrypoints, settings, and optional infra—you own every real implementation detail.

## What You Need To Build

1. **Ingestion pipeline** – subclass `BaseIngestionPipeline`, clean/chunk the corpus, attach metadata, and store the result inside the provided Postgres + `pgvector` instance (see `docker-compose.yml`). If you intentionally deviate from pgvector, that's fine, but please document why.
2. **Retrieval stack** – design embeddings, filters, rerankers, etc. No defaults are provided, but pgvector should be the primary backing store unless you have a compelling alternative :)
3. **Agent controller** – implement `BaseAgentController` with tool-aware planning and grounded responses. Tool hooks can be simple stubs, while the RAG + reasoning should be the focus.

## Repo Layout

```
.
├── Makefile                  # common commands (data/ingest/agent/eval/compose)
├── docker-compose.yml        # optional pgvector service
├── pyproject.toml            # deps + linting/type settings
├── src/agentic_rag/          # interfaces + CLI wiring
└── scripts/download_dataset.py
```

Feel free to rearrange directories, swap tooling, or add services. Keep the provided CLI commands working so reviewers can reproduce your runs please 

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make compose-up        # start pgvector (requires Docker)
make data              # download Hugging Face corpus/jsonl splits
make ingest            # runs your ingestion pipeline
make agent             # invokes your agent controller
make eval              # optional evaluator hook (implement if useful)
```

Stop the pgvector service with `make compose-down`. If the dataset needs auth, export `HF_TOKEN` before `make data`.

## Configuration

Point the CLI to your implementations via env vars (or `.env`):

```bash
export AGENTIC_RAG_INGESTION_CLASS="my_pkg.ingestion.Pipeline"
export AGENTIC_RAG_AGENT_CONTROLLER_CLASS="my_pkg.agent.Controller"
export AGENTIC_RAG_EVALUATOR_CLASS="my_pkg.eval.Runner"  # optional
```

## Delivering Your Work

- Share a repo link (public or private) with instructions for `make data`, `make ingest`, `make agent`, and any extras you added.
- Briefly describe the architecture and the trade-offs you made (models chosen, pgvector schema/indexing strategy, agent loop, limitations). If you did not use pgvector, call out the replacement and rationale explicitly.
- Everything else, such as tests, dashboards, fancy tool mocks is at your discretion.

That’s it! Good luck and I hope you will have fun :) 
