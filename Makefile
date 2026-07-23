.PHONY: setup fmt lint test clean ci-status ci-watch acquire bronze

setup:
	uv sync --extra dev

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

check: fmt lint test

ci-status:
	gh run list --workflow ci.yml --limit 5

ci-watch:
	gh run watch $$(gh run list --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId')

acquire:
	uv run python -m circuitnet_congestion.data.acquire --config configs/data.yaml

bronze:
	uv run python -m circuitnet_congestion.data.bronze --config configs/data.yaml