VENV ?= .venv
DATA_DIR ?= data/raw
PROCESSED_DIR ?= data/processed

ifeq ($(OS),Windows_NT)
    PYTHON ?= python
    ACTIVATE = $(VENV)\Scripts\activate
    PYTHON_VENV = $(VENV)\Scripts\python.exe
    PIP_VENV = $(VENV)\Scripts\pip.exe
else
    PYTHON ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)
    ACTIVATE = source $(VENV)/bin/activate
    PYTHON_VENV = $(VENV)/bin/python
    PIP_VENV = $(VENV)/bin/pip
endif

.PHONY: help venv install env data ingest agent server eval test clean compose-up compose-down

help:
	@echo "Available targets:"
	@echo "  make venv        # create virtualenv"
	@echo "  make install     # install deps"
	@echo "  make env         # create .env from .env.example"
	@echo "  make data        # download dataset into $(DATA_DIR)"
	@echo "  make ingest      # run ingestion pipeline"
	@echo "  make agent       # launch your agent controller"
	@echo "  make server      # start FastAPI server"
	@echo "  make eval        # run retrieval/agent evals"
	@echo "  make test        # run pytest"
	@echo "  make compose-up  # start optional pgvector stack"
	@echo "  make compose-down# stop optional pgvector stack"

venv:
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV) $(PYTHON) -m venv $(VENV)
else
	@if [ ! -d $(VENV) ]; then $(PYTHON) -m venv $(VENV); fi
endif

install: venv
	$(PYTHON_VENV) -m pip install -U pip || true
	$(PYTHON_VENV) -m pip install -r requirements.txt

env:
ifeq ($(OS),Windows_NT)
	@if not exist .env copy .env.example .env
else
	@if [ ! -f .env ]; then cp .env.example .env; fi
endif

data: install
	$(PYTHON_VENV) scripts/download_dataset.py --output $(DATA_DIR)

ingest: install
	$(PYTHON_VENV) -m agentic_rag.cli ingest --raw-dir $(DATA_DIR) --output-dir $(PROCESSED_DIR)

agent: install
	$(PYTHON_VENV) -m agentic_rag.cli agent

server: install
	$(PYTHON_VENV) -m agentic_rag.cli serve-api

_eval_cmd = $(PYTHON_VENV) -m agentic_rag.cli evaluate

eval: install
	$(_eval_cmd)

test: install
	$(PYTHON_VENV) -m pytest -q

compose-up:
	docker compose up -d vectorstore

compose-down:
	docker compose down

clean:
	rm -rf $(VENV) $(DATA_DIR) $(PROCESSED_DIR)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
