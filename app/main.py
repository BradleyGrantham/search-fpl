"""FPL player search: serves the sprite page plus a small JSON API."""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app import limits, nl_search
from app.ingest import DB_PATH, ingest, normalise

load_dotenv()

ROOT = Path(__file__).parent.parent
STATIC_DIR = ROOT / "static"
SEARCH_LOG = ROOT / "logs" / "searches.jsonl"
PUBLIC_COLUMNS = [
    "id", "code", "web_name", "first_name", "second_name", "team", "team_code",
    "position", "price", "total_points",
]
MAX_QUERY_CHARS = 200
REFRESH_HOURS = float(os.getenv("REFRESH_HOURS", "6"))

log = logging.getLogger("uvicorn.error")
_players: list[dict] = []


def load_players() -> None:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT {', '.join(PUBLIC_COLUMNS)}, name_search FROM players").fetchall()
    con.close()
    _players[:] = [dict(r) for r in rows]


def refresh_data() -> None:
    """Re-pull FPL data. ingest() swaps the db file atomically, so searches keep working meanwhile."""
    ingest()
    load_players()
    nl_search.refresh_prompt()  # the prompt includes the current gameweek
    log.info("Refreshed FPL data (%d players)", len(_players))


async def refresh_loop() -> None:
    while True:
        age_hours = (time.time() - DB_PATH.stat().st_mtime) / 3600
        await asyncio.sleep(max(0, REFRESH_HOURS - age_hours) * 3600)
        try:
            await run_in_threadpool(refresh_data)
        except Exception:
            log.exception("FPL data refresh failed; retrying in 15 minutes")
            await asyncio.sleep(15 * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not DB_PATH.exists():
        log.info("No database found, pulling data from FPL...")
        ingest()
    load_players()
    threading.Thread(target=nl_search.warm_up, daemon=True).start()
    refresher = asyncio.create_task(refresh_loop())
    yield
    refresher.cancel()


# No /docs or /openapi.json on the public site.
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def search_players(query: str) -> list[int]:
    """Instant name search: every word must appear in the player's name."""
    q = normalise(query)
    if not q:
        return []
    terms = q.split()
    hits = [p for p in _players if all(t in p["name_search"] for t in terms)]
    # Players whose display name starts with the query float to the top, then by points.
    hits.sort(key=lambda p: (not normalise(p["web_name"]).startswith(q), -p["total_points"]))
    return [p["id"] for p in hits]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "players": len(_players)}


@app.get("/api/players")
def players():
    return [{k: p[k] for k in PUBLIC_COLUMNS} for p in _players]


@app.get("/api/search")
def search(q: str = ""):
    return {"query": q, "ids": search_players(q[:MAX_QUERY_CHARS])}


@app.get("/api/ask")
async def ask(q: str, request: Request):
    q = q.strip()
    if not q or len(q) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Keep questions under {MAX_QUERY_CHARS} characters.")
    # In production the app listens on 127.0.0.1 behind a Cloudflare Tunnel, so only
    # cloudflared can connect and its CF-Connecting-IP header is the visitor's real IP.
    client = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")
    refusal = limits.check(client)
    if refusal:
        raise HTTPException(status_code=429, detail=refusal)

    try:
        result = await run_in_threadpool(nl_search.ask, q)
    except Exception as err:
        # Details go to the server log, not to visitors.
        log.exception("ask failed for %r", q)
        raise HTTPException(status_code=500, detail="Search failed. Try again, or rephrase it.") from err

    SEARCH_LOG.parent.mkdir(exist_ok=True)
    with SEARCH_LOG.open("a") as f:
        entry = {"ts": time.time(), "q": q, "n": len(result["ids"])}
        entry.update({k: result[k] for k in ("sql", "model_ms", "total_ms", "retried")})
        f.write(json.dumps(entry) + "\n")
    log.info("ask %r -> %d players, model %dms, total %dms",
             q, len(result["ids"]), result["model_ms"], result["total_ms"])
    return {"query": q, **result}
