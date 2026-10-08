"""Natural-language search: the prompt, and safe execution of the model's SQL.

Prompt text and stat labels match the server version (app/nl_search.py).
"""

import re
import sqlite3

MAX_ROWS = 10
# Stop any query after this much SQLite work. Real searches use at most ~250k operations;
# an operation budget (unlike a time limit) works on Workers, where the clock is frozen
# while code is running.
MAX_SQLITE_OPS = 5_000_000
_OPS_PER_TICK = 10_000

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


def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def build_system_prompt(con: sqlite3.Connection) -> str:
    teams = con.execute("SELECT short_name, name FROM teams ORDER BY name").fetchall()
    row = con.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
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


def open_read_only(db: bytes) -> sqlite3.Connection:
    """Load the snapshot into memory, then only ever allow reads on it."""
    con = sqlite3.connect(":memory:")
    con.deserialize(db)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
    return con


def run_sql(con: sqlite3.Connection, sql: str) -> tuple[list[int], list]:
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    ticks = 0

    def budget():
        nonlocal ticks
        ticks += 1
        return int(ticks * _OPS_PER_TICK > MAX_SQLITE_OPS)  # non-zero aborts the query

    con.set_progress_handler(budget, _OPS_PER_TICK)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        has_stat = cur.description is not None and len(cur.description) > 1
    finally:
        con.set_progress_handler(None, 0)
    return [int(r[0]) for r in rows], [r[1] if has_stat else None for r in rows]
