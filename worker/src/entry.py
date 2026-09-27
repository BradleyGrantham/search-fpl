"""FPL search API on Cloudflare Workers.

GET /players          every player (for the sprites)
GET /search?q=        instant name search
GET /ask?q=           natural-language search: gpt-oss-120b on Cerebras writes SQL, run read-only
GET /health

A cron trigger rebuilds the FPL snapshot into KV every 6 hours; each Worker instance keeps
it in memory as a read-only SQLite database.
"""

import json
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from js import AbortSignal, Object, fetch
from pyodide.ffi import to_js
from workers import Response, WorkerEntrypoint

import fpl_data
import nl

MAX_QUERY_CHARS = 120  # matches maxlength on the search box
SNAPSHOT_KEY = "fpl.db"
RELOAD_AFTER_S = 600  # re-read KV at most every 10 minutes per instance
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_TIMEOUT_MS = 8000
GENERIC_ERROR = "Search failed. Try again, or rephrase it."

# Per-instance cache of the loaded snapshot.
_state: dict | None = None


def js_obj(d: dict):
    return to_js(d, dict_converter=Object.fromEntries)


async def fetch_json(url: str):
    resp = await fetch(url)
    if not resp.ok:
        raise RuntimeError(f"GET {url} -> {resp.status}")
    return json.loads(await resp.text())


# --- snapshot -----------------------------------------------------------------

async def refresh_snapshot(env) -> bytes:
    bootstrap = await fetch_json(fpl_data.BOOTSTRAP_URL)
    fixtures = await fetch_json(fpl_data.FIXTURES_URL)
    db = fpl_data.build_db(bootstrap, fixtures)
    await env.FPL_DATA.put(SNAPSHOT_KEY, to_js(db))
    print(json.dumps({"event": "refresh", "bytes": len(db), "players": len(bootstrap["elements"])}))
    return db


def load_state(db: bytes) -> dict:
    con = nl.open_read_only(db)
    con.row_factory = sqlite3.Row
    players = [dict(r) for r in con.execute(
        f"SELECT {', '.join(fpl_data.PUBLIC_COLUMNS)}, name_search FROM players")]
    con.row_factory = None
    return {
        "con": con,
        "prompt": nl.build_system_prompt(con),
        "players": players,
        "players_json": json.dumps([{k: p[k] for k in fpl_data.PUBLIC_COLUMNS} for p in players]),
        "loaded_at": time.time(),
    }


async def get_state(env) -> dict:
    global _state
    if _state and time.time() - _state["loaded_at"] < RELOAD_AFTER_S:
        return _state
    buf = await env.FPL_DATA.get(SNAPSHOT_KEY, "arrayBuffer")
    if buf is None:
        db = await refresh_snapshot(env)  # first ever run
    else:
        # KV hands back a memoryview (or a JS ArrayBuffer proxy, depending on the runtime).
        db = buf.to_bytes() if hasattr(buf, "to_bytes") else bytes(buf)
    _state = load_state(db)
    return _state


# --- search -------------------------------------------------------------------

def search_players(players: list[dict], query: str) -> list[int]:
    """Instant name search: every word must appear in the player's name."""
    q = fpl_data.normalise(query)
    if not q:
        return []
    terms = q.split()
    hits = [p for p in players if all(t in p["name_search"] for t in terms)]
    # Players whose display name starts with the query float to the top, then by points.
    hits.sort(key=lambda p: (not fpl_data.normalise(p["web_name"]).startswith(q), -p["total_points"]))
    return [p["id"] for p in hits]


async def generate_sql(env, prompt: str, question: str, failed_sql=None, error=None) -> str:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    if failed_sql:
        messages += [
            {"role": "assistant", "content": failed_sql},
            {"role": "user", "content": f"That failed with: {error}\nFix it. Output only SQL."},
        ]
    body = json.dumps({
        "model": nl.MODEL,
        "messages": messages,
        "reasoning_effort": "low",
        "max_completion_tokens": 1024,
        "temperature": 0,
    })
    for attempt in range(2):  # one retry for Cerebras hiccups (5xx, 429, network)
        try:
            resp = await fetch(CEREBRAS_URL, js_obj({
                "method": "POST",
                "headers": js_obj({
                    "Authorization": f"Bearer {env.CEREBRAS_API_KEY}",
                    "Content-Type": "application/json",
                }),
                "body": body,
                "signal": AbortSignal.timeout(CEREBRAS_TIMEOUT_MS),
            }))
        except Exception as err:  # timeout or network error
            if attempt == 0:
                continue
            raise RuntimeError(f"Cerebras request failed: {err}") from err
        text = await resp.text()
        if resp.ok:
            return nl.clean_sql(json.loads(text)["choices"][0]["message"]["content"] or "")
        if attempt == 0 and (resp.status == 429 or resp.status >= 500):
            continue
        raise RuntimeError(f"Cerebras {resp.status}: {text[:300]}")


async def within_daily_cap(env) -> bool:
    """Approximate global cap on AI searches per day (KV is eventually consistent)."""
    limit = int(getattr(env, "ASK_DAILY_LIMIT", "2000") or 2000)
    key = "asks:" + datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = int(await env.FPL_DATA.get(key) or 0)
    if count >= limit:
        return False
    try:
        await env.FPL_DATA.put(key, str(count + 1), js_obj({"expirationTtl": 2 * 86400}))
    except Exception:
        pass  # KV allows ~1 write/second per key; a missed increment only loosens the cap
    return True


async def ask(env, request, question: str):
    question = question.strip()
    if not question or len(question) > MAX_QUERY_CHARS:
        return 400, {"detail": f"Keep questions under {MAX_QUERY_CHARS} characters."}
    client = request.headers.get("cf-connecting-ip") or "unknown"
    outcome = await env.ASK_LIMITER.limit(js_obj({"key": client}))
    if not outcome.success:
        return 429, {"detail": "Too many AI searches in a row. Wait a minute and try again."}
    if not await within_daily_cap(env):
        return 429, {"detail": "AI search has hit its daily limit. Name search still works; try again tomorrow."}

    state = await get_state(env)
    t0 = time.time()
    sql = await generate_sql(env, state["prompt"], question)
    retried = False
    try:
        ids, stats = nl.run_sql(state["con"], sql)
    except (sqlite3.Error, ValueError) as err:
        retried = True
        sql = await generate_sql(env, state["prompt"], question, failed_sql=sql, error=str(err))
        ids, stats = nl.run_sql(state["con"], sql)
    total_ms = round((time.time() - t0) * 1000)  # wall time across the model call(s)

    print(json.dumps({"event": "ask", "q": question, "n": len(ids), "sql": sql,
                      "total_ms": total_ms, "retried": retried}))
    result = {"query": question, "ids": ids, "stats": stats, "stat_label": nl.stat_label(sql),
              "retried": retried, "total_ms": total_ms}
    if getattr(env, "SHOW_SQL", "") in ("1", "true", "yes"):
        result["sql"] = sql
    return 200, result


# --- HTTP -----------------------------------------------------------------------

def cors_headers(env, request) -> dict:
    origin = request.headers.get("origin")
    allowed = [o.strip() for o in str(getattr(env, "ALLOWED_ORIGINS", "")).split(",") if o.strip()]
    if origin and origin in allowed:
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {"Vary": "Origin"}


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        env = self.env
        cors = cors_headers(env, request)

        def reply(status: int, payload, extra: dict | None = None):
            body = payload if isinstance(payload, str) else json.dumps(payload)
            headers = {"Content-Type": "application/json", **cors, **(extra or {})}
            return Response(body, status=status, headers=headers)

        if request.method == "OPTIONS":
            return Response("", status=204, headers={
                **cors, "Access-Control-Allow-Methods": "GET", "Access-Control-Max-Age": "86400"})
        if request.method != "GET":
            return reply(405, {"detail": "Method not allowed"})

        url = urlparse(request.url)
        path = url.path.rstrip("/") or "/"
        q = parse_qs(url.query).get("q", [""])[0]
        try:
            if path == "/health":
                state = await get_state(env)
                return reply(200, {"ok": True, "players": len(state["players"])})
            if path == "/players":
                state = await get_state(env)
                return reply(200, state["players_json"], {"Cache-Control": "public, max-age=300"})
            if path == "/search":
                state = await get_state(env)
                q = q[:MAX_QUERY_CHARS]
                return reply(200, {"query": q, "ids": search_players(state["players"], q)})
            if path == "/ask":
                status, payload = await ask(env, request, q)
                return reply(status, payload)
            return reply(404, {"detail": "Not found"})
        except Exception:
            # Details go to the Worker logs, not to visitors.
            print(json.dumps({"event": "error", "path": path, "q": q[:200], "trace": traceback.format_exc()}))
            return reply(500, {"detail": GENERIC_ERROR})

    async def scheduled(self, *args):
        global _state
        db = await refresh_snapshot(self.env)
        _state = load_state(db)
