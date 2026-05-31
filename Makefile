.PHONY: install train test app lint

install:
	pip install -e ".[dev]"

train:
	python -m src.models.train

test:
	pytest -q

app:
	streamlit run app/Home.py

lint:
	ruff check src app tests
	mypy src
