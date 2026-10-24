# pywrangler's Pyodide launcher breaks on paths with spaces, and uv's default Python
# directory on macOS is "~/Library/Application Support/uv/python".
export UV_PYTHON_INSTALL_DIR := $(HOME)/.local/share/uv/python

.PHONY: dev deploy logs bench
dev:     ## local server at http://localhost:8787 (GET /cdn-cgi/handler/scheduled runs the refresh)
	uv run pywrangler dev --test-scheduled
deploy:
	uv run pywrangler deploy
logs:
	npx wrangler tail fpl-api
bench:   ## time /ask over bench/questions.txt (needs ANTHROPIC_API_KEY in .env)
	uv run python bench/run.py --reps 5 --label baseline
	uv run python bench/report.py bench/results/baseline.jsonl --write-ground-truth
