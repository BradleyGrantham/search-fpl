# FPL Search

Type a player's name, or ask a question in plain English, and the matching players are pulled out of the pile.

Live at https://bradleygrantham.com/fpl/.

## How it works

The API is a Cloudflare Python Worker (`fpl-api`):

- Every 6 hours it pulls the FPL data into a small SQLite database and stores it in Workers KV.
- Name search runs in memory on that data.
- AI search asks Claude Haiku to turn the question into one SQLite `SELECT`. That query runs on a read-only, in-memory copy of the database, with an authorizer that only allows reads and a cap on how long a query can run and how many rows it returns.

On bradleygrantham.com the API is routed at `/fpl/api/*`, so the page calls it on its own origin.

## Files

- `src/entry.py`: the Worker. Routes (`/players`, `/search`, `/ask`, `/health`), CORS, rate limits and the 6-hourly refresh.
- `src/fpl_data.py`: builds the SQLite database from the FPL API's JSON.
- `src/nl.py`: the prompt (rules, data dictionary, examples) and safe execution of the model's SQL.
- `src/llm.py`: the Claude call, with prompt caching.
- `static/index.html`: the page. Typing does name search; if no names match, pausing (or pressing Enter) asks the AI.
- `wrangler.jsonc`: Worker config (route, cron, KV, rate limits, allowed origins).

## Running it

Needs [uv](https://docs.astral.sh/uv/) and Node (for Wrangler).

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .dev.vars   # once, for local dev
make dev       # http://localhost:8787; GET /cdn-cgi/handler/scheduled builds the data
make deploy    # set the key first: npx wrangler secret put ANTHROPIC_API_KEY
make logs
```

To use the page against your local API, set `<meta name="fpl-api">` in `static/index.html` to `http://localhost:8787`.
