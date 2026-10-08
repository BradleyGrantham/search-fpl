"""Natural-language search: question -> SQL -> player ids.

The SQL is written by gpt-oss-120b on Cerebras (default), or by Claude Haiku 5.5 when
LLM_PROVIDER=anthropic (see app/claude_sql.py).
"""

import os
import re
import sqlite3
import time
from pathlib import Path

from openai import OpenAI

from app import claude_sql
from app.ingest import DB_PATH

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()  # "cerebras" | "anthropic"
MODEL = "gpt-oss-120b"
MAX_ROWS = 10
QUERY_TIMEOUT_S = 2.0

_client: OpenAI | None = None
_system_prompt: str | None = None

DATA_DICTIONARY = """\
players: one row per FPL player, current season to date.
  id, web_name (display name e.g. 'Haaland'), first_name, second_name
  name_search: lowercase accent-free 'first second web_name' - use LIKE '%term%' to find a player by name
  team_id, team (short code e.g. 'ARS'), team_name (e.g. 'Arsenal')
  position: 'GKP' | 'DEF' | 'MID' | 'FWD'
  price: current price in £m (e.g. 5.5)
  total_points: FPL points this season; event_points: points in the current gameweek
  form: average points per match over the last 30 days
  points_per_game, ep_next (FPL's expected points next gameweek)
  selected_by_percent: ownership %, 0-100 ('differential' usually means < 10)
  minutes, starts, goals_scored, assists, clean_sheets, goals_conceded, saves, bonus, bps,
  yellow_cards, red_cards, penalties_missed
  expected_goals (xG), expected_assists (xA), expected_goal_involvements (xGI), expected_goals_conceded (xGC),
  expected_goals_per_90, expected_assists_per_90, expected_goal_involvements_per_90
  defensive_contribution, defensive_contribution_per_90 (CBIT points stat), ict_index
  status: 'a' available, 'd' doubtful, 'i' injured, 's' suspended, 'u' unavailable/left club, 'n' not eligible
  chance_of_playing_next_round: 0-100, NULL means no concerns
  news: injury/availability text
  transfers_in_event, transfers_out_event: transfers this gameweek
  cost_change_start: price change since season start in £m
  penalties_order, direct_freekicks_order, corners_and_indirect_freekicks_order: 1 = first choice taker, NULL = not on set pieces
  birth_date: 'YYYY-MM-DD'
teams: id, code, name, short_name, strength_overall_home, strength_overall_away
fixtures: id, gameweek, kickoff_time, home_team_id, away_team_id, home_score, away_score,
  home_difficulty, away_difficulty (FDR 1 easy - 5 hard), finished (0/1)
gameweeks: id, deadline_time, is_current, is_next, finished
team_fixtures (view, one row per team per fixture): fixture_id, gameweek, kickoff_time, team_id,
  opponent_id, is_home, difficulty (FDR for that team, 1 easy - 5 hard), finished
"""

RULES = """\
Write ONE SQLite SELECT statement that answers the user's question about Fantasy Premier League players.
Output only the SQL. No markdown, no explanation.

The result MUST have columns:
  1. id  (players.id)
  2. stat  (optional) - the single number most relevant to the question (e.g. price for 'cheap', expected_goals for 'xG'),
     rounded to 2dp. Omit it if nothing specific was asked about.
Order rows so the best answers come first, and end with LIMIT {max_rows}.
Unless the question is about injuries/unavailable players, exclude players with status 'u' or 'n'.
For recommendation-style questions (in form, differentials, easy fixtures, value, 'who should I buy'), only include
players who actually play (minutes >= 90 so far) and rank by the quality asked for, not by price.
'Cheap'/'budget' with no number is a price cap (GKP <= 4.5, DEF <= 4.5, MID <= 5.5, FWD <= 6.0), not a sort order.
'Premium' means price >= 9.5.
Per-90 stats are meaningless on tiny samples: whenever you filter or rank by a *_per_90 column, require minutes >= 270.
If the question is just a player's name, match it with name_search.
If the question cannot be answered from this data, output: SELECT id FROM players WHERE 0
"""

EXAMPLES = """\
Q: arsenal defenders under 5m
SELECT id, price AS stat FROM players WHERE team = 'ARS' AND position = 'DEF' AND price <= 5.0 AND status NOT IN ('u','n') ORDER BY total_points DESC LIMIT 10

Q: midfielders with the best xG per 90 who've played at least 300 mins
SELECT id, ROUND(expected_goals_per_90, 2) AS stat FROM players WHERE position = 'MID' AND minutes >= 300 AND status NOT IN ('u','n') ORDER BY expected_goals_per_90 DESC LIMIT 10

Q: cheap in-form midfielders
SELECT id, form AS stat FROM players WHERE position = 'MID' AND price <= 5.5 AND minutes >= 90 AND status NOT IN ('u','n') ORDER BY form DESC LIMIT 10

Q: forwards with easy fixtures over the next 3 gameweeks
SELECT p.id, ROUND(AVG(tf.difficulty), 2) AS stat FROM players p JOIN team_fixtures tf ON tf.team_id = p.team_id WHERE p.position = 'FWD' AND p.minutes >= 90 AND p.status NOT IN ('u','n') AND tf.gameweek BETWEEN {next_gw} AND {next_gw} + 2 GROUP BY p.id ORDER BY stat ASC, p.total_points DESC LIMIT 10
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Fail fast: a search box is useless after a few seconds (OpenAI's default is 10 min, 2 retries).
        _client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ["CEREBRAS_API_KEY"],
            timeout=5.0,
            max_retries=1,
        )
    return _client


def warm_up() -> None:
    """Build the prompt and get the model ready so the first search isn't slow."""
    global _system_prompt
    _system_prompt = build_system_prompt()
    if LLM_PROVIDER == "anthropic":
        claude_sql.warm_up(_system_prompt)  # writes the prompt cache
    else:
        _get_client().models.list()  # opens the HTTPS connection


def refresh_prompt() -> None:
    """Rebuild the prompt after a data refresh (the current gameweek may have moved on)."""
    global _system_prompt
    old, _system_prompt = _system_prompt, build_system_prompt()
    if LLM_PROVIDER == "anthropic" and _system_prompt != old:
        claude_sql.warm_up(_system_prompt)  # new prompt text means a new cache entry


def _read_only_connection() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)

    # Belt and braces on top of mode=ro: only allow reads.
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

    def authorizer(action, *_):
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    con.set_authorizer(authorizer)
    return con


def build_system_prompt() -> str:
    con = sqlite3.connect(DB_PATH)
    teams = con.execute("SELECT short_name, name FROM teams ORDER BY name").fetchall()
    row = con.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
    con.close()
    next_gw = row[0] if row else 38
    team_list = ", ".join(f"{short}={name}" for short, name in teams)
    return "\n".join([
        RULES.format(max_rows=MAX_ROWS),
        f"The next gameweek is {next_gw}; the current one is {next_gw - 1}.",
        f"Teams (short_name=name): {team_list}",
        "",
        "Schema:",
        DATA_DICTIONARY,
        "Examples:",
        EXAMPLES.format(next_gw=next_gw),
    ])


STAT_LABELS = {
    "price": "£", "form": "form", "total_points": "pts", "event_points": "GW pts",
    "points_per_game": "ppg", "ep_next": "xPts", "selected_by_percent": "own%",
    "minutes": "mins", "goals_scored": "goals", "assists": "assists", "clean_sheets": "CS",
    "saves": "saves", "bonus": "bonus", "expected_goals": "xG", "expected_assists": "xA",
    "expected_goal_involvements": "xGI", "expected_goals_conceded": "xGC",
    "expected_goals_per_90": "xG/90", "expected_assists_per_90": "xA/90",
    "expected_goal_involvements_per_90": "xGI/90", "defensive_contribution": "DC",
    "defensive_contribution_per_90": "DC/90", "ict_index": "ICT", "difficulty": "FDR",
    "chance_of_playing_next_round": "chance%", "penalties_order": "pen",
    "transfers_in_event": "in", "transfers_out_event": "out", "cost_change_start": "£ change",
    "birth_date": "age",
}


def stat_label(sql: str) -> str | None:
    """Best-effort short label for the `stat` column, from the last column name before 'AS stat'."""
    match = re.search(r"([\s\S]*?)\bAS\s+stat\b", sql, re.IGNORECASE)
    if not match:
        return None
    columns = re.findall(r"[a-z_][a-z0-9_]*", match.group(1).split(",", 1)[-1])
    known = {c for c in columns if c in STAT_LABELS}
    # Only label simple stats; a combination like total_points / price gets no label.
    return STAT_LABELS[known.pop()] if len(known) == 1 else None


def _clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def _generate_sql(question: str, failed_sql: str | None = None, error: str | None = None):
    """Returns (sql, usage); usage is only reported for Claude."""
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = build_system_prompt()
    if LLM_PROVIDER == "anthropic":
        text, usage = claude_sql.generate_sql(_system_prompt, question, failed_sql, error)
        return _clean_sql(text), usage
    messages = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": question},
    ]
    if failed_sql:
        messages += [
            {"role": "assistant", "content": failed_sql},
            {"role": "user", "content": f"That failed with: {error}\nFix it. Output only SQL."},
        ]
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort="low",
        max_completion_tokens=1024,
        temperature=0,
    )
    return _clean_sql(response.choices[0].message.content or ""), None


def _run_sql(sql: str) -> tuple[list[int], list[float | None]]:
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    con = _read_only_connection()
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        has_stat = cur.description and len(cur.description) > 1
    finally:
        con.close()
    ids = [int(r[0]) for r in rows]
    stats = [r[1] if has_stat else None for r in rows]
    return ids, stats


def ask(question: str) -> dict:
    """Answer a natural-language question. Retries once, feeding the SQL error back to the model."""
    t0 = time.perf_counter()
    sql, usage = _generate_sql(question)
    model_ms = (time.perf_counter() - t0) * 1000
    retried = False
    try:
        ids, stats = _run_sql(sql)
    except (sqlite3.Error, ValueError) as err:
        retried = True
        t1 = time.perf_counter()
        sql, retry_usage = _generate_sql(question, failed_sql=sql, error=str(err))
        if usage and retry_usage:
            usage = {k: usage[k] + retry_usage[k] for k in usage}
        model_ms += (time.perf_counter() - t1) * 1000
        ids, stats = _run_sql(sql)
    return {
        "ids": ids,
        "stats": stats,
        "stat_label": stat_label(sql),
        "sql": sql,
        "retried": retried,
        "model": claude_sql.MODEL if LLM_PROVIDER == "anthropic" else MODEL,
        "usage": usage,  # Claude only: input / cache_read / cache_write / output tokens
        "model_ms": round(model_ms),
        "total_ms": round((time.perf_counter() - t0) * 1000),
    }
