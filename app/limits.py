"""In-memory rate limits for the AI endpoint, which costs money per call.

Single-process only (fine for one uvicorn worker); limits reset on restart.
"""

import os
import time
from collections import defaultdict, deque

PER_MINUTE = int(os.getenv("ASK_LIMIT_PER_MINUTE", "20"))
PER_DAY_PER_CLIENT = int(os.getenv("ASK_LIMIT_PER_DAY_PER_CLIENT", "300"))
PER_DAY_TOTAL = int(os.getenv("ASK_LIMIT_PER_DAY_TOTAL", "2000"))

_recent: dict[str, deque] = defaultdict(deque)   # client -> timestamps in the last minute
_daily: dict[str, int] = defaultdict(int)         # client -> count today
_day = time.strftime("%Y-%m-%d")
_total_today = 0


def check(client: str) -> str | None:
    """Record an AI search for this client. Returns a message if it should be refused."""
    global _day, _total_today
    today = time.strftime("%Y-%m-%d")
    if today != _day:
        _day, _total_today = today, 0
        _daily.clear()

    if _total_today >= PER_DAY_TOTAL:
        return "AI search has hit its daily limit. Name search still works; try again tomorrow."
    if _daily[client] >= PER_DAY_PER_CLIENT:
        return "You've reached today's limit for AI searches. Name search still works."

    now = time.monotonic()
    recent = _recent[client]
    while recent and now - recent[0] > 60:
        recent.popleft()
    if len(recent) >= PER_MINUTE:
        return "Too many AI searches in a row. Wait a minute and try again."

    recent.append(now)
    _daily[client] += 1
    _total_today += 1
    return None
