.PHONY: install check test browser-test run data-download data-normalize data-index data-evaluate docker-build

install:
	uv sync --all-groups

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

test:
	uv run pytest --cov-fail-under=85

browser-test:
	uv run playwright install chromium
	uv run python tests/browser_smoke.py

run:
	uv run faq-chatbot

data-download:
	uv run faq-index download

data-normalize:
	uv run faq-index normalize data/raw/smartstore_faq.pkl --output data/faqs.jsonl

data-index:
	uv run faq-index build --input data/faqs.jsonl

data-evaluate:
	uv run faq-index evaluate data/eval/smartstore_eval.json

docker-build:
	docker build -t faq-chatbot:local .
