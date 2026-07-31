.PHONY: setup fmt lint test check clean ci-status ci-watch acquire bronze silver gold report report-check train board probe

CONFIG ?= configs/unet_a.yaml

setup:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		uv sync --extra dev --extra cuda; \
	else \
		uv sync --extra dev --extra cpu; \
	fi

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

silver:
	uv run python -m circuitnet_congestion.data.silver --config configs/data.yaml

gold:
	uv run python -m circuitnet_congestion.data.gold --config configs/data.yaml

report:
	uv run python -m analysis.render_report

# Re-render from committed artefacts and fail if anything moved. This is the
# repository's central claim made testable: a clone with no data layer and no
# accelerator regenerates every document byte-for-byte.
report-check:
	uv run python -m analysis.render_report
	git diff --exit-code -- README.md docs/*.md

train:
	uv run python -m circuitnet_congestion.training.train --config $(CONFIG)

board:
	uv run tensorboard --logdir runs --port 6006 --bind_all

probe:
	uv run python -m analysis.probe_target_stats
	uv run python -m analysis.probe_selection_gap
	uv run python -m analysis.probe_throughput
	uv run python -m analysis.probe_precision
	uv run python -m analysis.probe_optimisation

probe-test:
	uv run python -m analysis.probe_test_eval