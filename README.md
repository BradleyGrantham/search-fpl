# FPL Search

Type a player's name, or ask a question in plain English, and the matching players are pulled out of the pile.

```bash
echo "CEREBRAS_API_KEY=..." > .env   # once
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000. URL shortcuts: `/?q=haaland` (name search), `/?ask=cheap+in-form+defenders` (AI search).

- `app/ingest.py`: pulls FPL data into `fpl.db` (SQLite). Runs automatically if the db is missing; refresh with `uv run python -m app.ingest`.
- `app/nl_search.py`: question -> SQL with gpt-oss-120b on Cerebras, run on a read-only, SELECT-only connection. The prompt (rules, data dictionary, examples) lives here.
- `app/main.py`: FastAPI. `/api/search` is the instant name search, `/api/ask` is the AI search. Every AI search is logged to `logs/searches.jsonl` with its SQL and timings.
- `static/index.html`: the page. Typing does name search; if no names match, pausing (or pressing Enter) asks the AI.
