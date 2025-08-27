VENV=.venv
PYTHON=$(VENV)/bin/python

venv: requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn ui.backend.server:app --reload --port 8000

loader:
	$(PYTHON) data/fiqa_loader.py

embed:
	$(PYTHON) embeddings/embed.py --input data/sample.csv --output embeddings/fiqa.parquet
