# FPL Search

Type a player's name, or ask a question in plain English, and the matching players are pulled out of the pile.

Live at https://bradleygrantham.com/fpl/.

## How it works

The API is a Cloudflare Python Worker (`fpl-api`):

- Every 6 hours it pulls the FPL data into a small SQLite database and stores it in Workers KV.
- Name search runs in memory on that data.
- AI search asks Claude Haiku to turn the question into one SQLite `SELECT`. That query runs on a read-only, in-memory copy of the database, with an authorizer that only allows reads and a cap on how long a query can run and how many rows it returns.
- The model queries a version of the data built for short SQL (lowercase, short column names, precomputed fixture difficulty and value columns), since the SQL it writes is most of the time a search takes that we can change. `bench/` measures it; a search typically comes back in ~0.8 s, ~0.65 s of which is the wait for Haiku's first token.

On bradleygrantham.com the API is routed at `/fpl/api/*`, so the page calls it on its own origin.

## Files

- `src/entry.py`: the Worker. Routes (`/players`, `/search`, `/ask`, `/health`), CORS, rate limits and the 6-hourly refresh.
- `src/fpl_data.py`: builds the SQLite database from the FPL API's JSON.
- `src/nl.py`: the prompt, the tables the model queries, and safe execution of its SQL.
- `src/llm.py`: the Claude call, with prompt caching.
- `static/index.html`: the page. Typing does name search; if no names match, pausing (or pressing Enter) asks the AI.
- `wrangler.jsonc`: Worker config (route, cron, KV, rate limits, allowed origins).
- `bench/`: the AI search benchmark: 111 questions (`questions.txt`), the old prompt's answers as a reference (`ground_truth.jsonl`), A/B runs of prompt variants (`run.py`, `variants.py`) and reports (`report.py`). Results go in `bench/results/`, which isn't committed.

## Running it

Needs [uv](https://docs.astral.sh/uv/) and Node (for Wrangler).

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .dev.vars   # once, for local dev
make dev       # http://localhost:8787; GET /cdn-cgi/handler/scheduled builds the data
make deploy    # set the key first: npx wrangler secret put ANTHROPIC_API_KEY
make logs
```

To use the page against your local API, set `<meta name="fpl-api">` in `static/index.html` to `http://localhost:8787`.
