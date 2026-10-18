# FPL Search

Type a player's name, or ask a question in plain English, and the matching players are pulled out of the pile.

Live at https://bradleygrantham.com/fpl/.

## How it runs

- `static/index.html`: the page. Typing does name search; if no names match, pausing (or pressing Enter) asks the AI.
- `worker/`: the live API, a Cloudflare Python Worker (`fpl-api`). It rebuilds an SQLite snapshot of the FPL data every 6 hours into KV, does name search in memory, and turns questions into SQL with Claude Haiku. Routed at `bradleygrantham.com/fpl/api/*`.
  ```bash
  cd worker
  npx wrangler secret put ANTHROPIC_API_KEY   # once
  make dev      # http://localhost:8787
  make deploy
  ```

## Local FastAPI version

The original version of the API, handy for working on the prompt.

```bash
echo "ANTHROPIC_API_KEY=..." > .env   # once (or LLM_PROVIDER=cerebras + CEREBRAS_API_KEY)
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000. URL shortcuts: `/?q=haaland` (name search), `/?ask=cheap+in-form+defenders` (AI search).

- `app/ingest.py`: pulls FPL data into `fpl.db` (SQLite). Runs automatically if the db is missing; refresh with `uv run python -m app.ingest`.
- `app/nl_search.py`: question -> SQL (Claude Haiku by default, or gpt-oss-120b on Cerebras), run on a read-only, SELECT-only connection. The prompt (rules, data dictionary, examples) lives here.
- `app/main.py`: FastAPI. `/api/search` is the instant name search, `/api/ask` is the AI search. Every AI search is logged to `logs/searches.jsonl` with its SQL and timings.
