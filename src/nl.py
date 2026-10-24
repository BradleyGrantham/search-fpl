"""Natural-language search: the prompt, the tables the model queries, and safe execution of its SQL.

Most of a search's time is the model, and the part we control is how many tokens it writes, so
the prompt and tables are built for short SQL: lowercase keywords, short column names, and
precomputed columns (fixture difficulty, value, position flags) instead of joins and rules.
bench/ measures every change; see bench/variants.py for the experiments behind this one.
"""

import re
import sqlite3

MAX_ROWS = 10
# Stop any query after this much SQLite work. Real searches use at most ~250k operations;
# an operation budget (unlike a time limit) works on Workers, where the clock is frozen
# while code is running.
MAX_SQLITE_OPS = 5_000_000
_OPS_PER_TICK = 10_000

# The snapshot's own players table, renamed when the snapshot is opened so that `players` can be
# the model's view. /players and /search read this one.
SOURCE_PLAYERS = "fpl_players"

# Snapshot column -> the shorter name the model sees. Everything else keeps its name.
SHORT_NAMES = {
    "total_points": "pts", "event_points": "gw_pts", "points_per_game": "ppg", "ep_next": "xpts_next",
    "selected_by_percent": "own", "goals_scored": "goals", "goals_conceded": "conceded",
    "penalties_missed": "pens_missed", "expected_goals": "xg", "expected_assists": "xa",
    "expected_goal_involvements": "xgi", "expected_goals_conceded": "xgc",
    "expected_goals_per_90": "xg90", "expected_assists_per_90": "xa90",
    "expected_goal_involvements_per_90": "xgi90", "defensive_contribution": "dc",
    "defensive_contribution_per_90": "dc90", "chance_of_playing_next_round": "chance",
    "transfers_in_event": "t_in", "transfers_out_event": "t_out", "cost_change_start": "price_change",
    "penalties_order": "pen_order", "direct_freekicks_order": "fk_order",
    "corners_and_indirect_freekicks_order": "corner_order",
}
# Per-90 stats on a tiny sample are noise, so they are blank under 270 minutes.
PER_90 = {"expected_goals_per_90", "expected_assists_per_90", "expected_goal_involvements_per_90",
          "defensive_contribution_per_90"}
# 0/1 position flags; `gk` isn't in the prompt but the model sometimes writes it for 'gks'.
POSITION_FLAGS = {"gkp": "GKP", "def": "DEF", "mid": "MID", "fwd": "FWD", "gk": "GKP"}

PROMPT = """\
Turn the user's message into ONE lowercase SQLite select over the FPL tables below. Reply with SQL only, never prose.
Vague FPL questions (who to buy, captain, any tips) are recommendations. If players can't answer it (the weather, who wins the league, a poem): select id from players where 0
Rows are always players, never teams. A question about teams (worst defence, best attack, hardest fixtures) wants those teams' players.
Columns: id, then optionally the one number the question is about (e.g. price for "cheap"). Best first; first {max_rows} rows are used (no LIMIT/ROUND/alias).
players = available players; all_players also has those who left (status u/n). A name (even misspelt): all_players where name_search like '%name%'.
Recommendations (form, differentials, fixtures, value, who to buy): add regular and rank by that quality, not price. Plain lists and filters (a team's forwards, players under X, young players) don't need regular; rank them by pts.
Lowest/fewest of a stat: minutes >= 270.
under/over Xm = price <= X / >= X. cheap/budget = budget; premium = price >= 9.5; template = most owned.
Next gameweek {next_gw}; {played} played, so starts = {played} means every game. Teams: {teams}

players: id, web_name, name_search (lowercase 'first last web_name'), team (code), team_name, position (GKP/DEF/MID/FWD; flags gkp, def, mid, fwd),
price (£m), pts, gw_pts (this gameweek), form, ppg, xpts_next, own (%; differential < 10), minutes, starts, goals, assists,
clean_sheets, conceded, saves, bonus, bps, yellow_cards, red_cards, pens_missed, xg, xa, xgi, xgc, xg90, xa90, xgi90, dc, dc90
(per-90s NULL under 270 mins), ict_index, status (a ok, d doubtful, i injured, s suspended), chance (0-100; NULL or 100 = fit), news,
t_in, t_out (this gameweek), price_change (since start), pen_order, fk_order, corner_order (1 = first choice, NULL = none),
birth_date, regular (minutes >= 90), budget (cheap for position), pts_per_m, xg_diff (goals - xg), fixtures_next (2 double, 0 blank),
home_next, fdr_next, fdr_next3, fdr_next5 (avg difficulty next 1/3/5 GWs, 1 easy - 5 hard),
team_scored, team_conceded (the player's team's goals per finished game)
team_fixtures: team_id, gameweek, opponent_id, is_home, difficulty. teams: id, name, short_name

Q: arsenal defenders under 5m
select id, price from players where team = 'ARS' and def and price <= 5 order by pts desc
Q: cheap in-form midfielders
select id, form from players where mid and budget and regular order by 2 desc
Q: forwards with easy fixtures next 3 gameweeks
select id, fdr_next3 from players where fwd and regular order by 2, pts desc
Q: haalnd
select id from all_players where name_search like '%haaland%'
Q: worst defence
select id, team_conceded from players where (gkp or def) and regular order by 2 desc, pts desc
"""

STAT_LABELS = {
    "price": "£", "form": "form", "pts": "pts", "gw_pts": "GW pts", "ppg": "ppg", "xpts_next": "xPts",
    "own": "own%", "minutes": "mins", "starts": "starts", "goals": "goals", "assists": "assists", "clean_sheets": "CS",
    "saves": "saves", "bonus": "bonus", "xg": "xG", "xa": "xA", "xgi": "xGI", "xgc": "xGC",
    "xg90": "xG/90", "xa90": "xA/90", "xgi90": "xGI/90", "dc": "DC", "dc90": "DC/90", "ict_index": "ICT",
    "difficulty": "FDR", "chance": "chance%", "pen_order": "pen", "t_in": "in", "t_out": "out",
    "price_change": "£ change", "birth_date": "age", "pts_per_m": "pts/£m", "xg_diff": "G−xG",
    "fdr_next": "FDR", "fdr_next3": "FDR", "fdr_next5": "FDR", "fixtures_next": "fixtures",
    "team_scored": "team GF", "team_conceded": "team GA",
}
# Labels for a column returned as it is, where the label for an expression over it would be wrong.
BARE_STAT_LABELS = {"birth_date": "born"}


def stat_label(column: str) -> str | None:
    """Short label for the stat column, from its name (a column, or the expression SQLite names it by)."""
    if column in BARE_STAT_LABELS:
        return BARE_STAT_LABELS[column]
    known = {c for c in re.findall(r"[a-z_][a-z0-9_]*", column.lower()) if c in STAT_LABELS}
    # Only label simple stats; a combination like pts / price gets no label.
    return STAT_LABELS[known.pop()] if len(known) == 1 else None


def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def is_select(text: str) -> bool:
    return bool(re.match(r"^\s*(SELECT|WITH)\b", text, re.IGNORECASE))


def build_system_prompt(con: sqlite3.Connection) -> str:
    teams = con.execute("SELECT short_name, name FROM teams ORDER BY name").fetchall()
    row = con.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
    next_gw = row[0] if row else 38
    return PROMPT.format(max_rows=MAX_ROWS, next_gw=next_gw, played=next_gw - 1,
                         teams=", ".join(f"{short}={name}" for short, name in teams))


def _add_model_tables(con: sqlite3.Connection) -> None:
    """all_players: every player under the model's names, plus precomputed columns; players: a
    view of the ones still available."""
    columns = [r[1] for r in con.execute("PRAGMA table_info(players)")]
    select = [f"CASE WHEN p.minutes >= 270 THEN p.{c} END AS {SHORT_NAMES.get(c, c)}" if c in PER_90
              else f"p.{c} AS {SHORT_NAMES.get(c, c)}" for c in columns]
    (gw,) = con.execute("SELECT COALESCE((SELECT id FROM gameweeks WHERE is_next = 1), 38)").fetchone()
    select += [
        "p.minutes >= 90 AS regular",
        "ROUND(p.total_points / p.price, 2) AS pts_per_m",
        "ROUND(p.goals_scored - p.expected_goals, 2) AS xg_diff",
        "p.price <= CASE p.position WHEN 'MID' THEN 5.5 WHEN 'FWD' THEN 6.0 ELSE 4.5 END AS budget",
        "COALESCE(f1.n, 0) AS fixtures_next", "f1.home AS home_next",
        "f1.fdr AS fdr_next", "f3.fdr AS fdr_next3", "f5.fdr AS fdr_next5",
        "goals.scored AS team_scored", "goals.conceded AS team_conceded",
        *(f"p.position = '{position}' AS {flag}" for flag, position in POSITION_FLAGS.items()),
    ]
    # Each team's fixtures over the next 1, 3 and 5 gameweeks: how many, any at home, average difficulty.
    fixtures = ("(SELECT team_id, COUNT(*) AS n, MAX(is_home) AS home, ROUND(AVG(difficulty), 2) AS fdr "
                "FROM team_fixtures WHERE gameweek BETWEEN {first} AND {last} GROUP BY team_id)")
    joins = " ".join(f"LEFT JOIN {fixtures.format(first=gw, last=gw + n - 1)} f{n} ON f{n}.team_id = p.team_id"
                     for n in (1, 3, 5))
    # Each team's goals scored and conceded per finished game, so team questions come back as players.
    joins += (" LEFT JOIN (SELECT team_id, ROUND(AVG(scored), 2) AS scored, ROUND(AVG(conceded), 2) AS conceded"
              " FROM (SELECT home_team_id AS team_id, home_score AS scored, away_score AS conceded"
              " FROM fixtures WHERE finished = 1 UNION ALL SELECT away_team_id, away_score, home_score"
              " FROM fixtures WHERE finished = 1) GROUP BY team_id) goals ON goals.team_id = p.team_id")
    con.executescript(
        f"ALTER TABLE players RENAME TO {SOURCE_PLAYERS};"
        f"CREATE TABLE all_players AS SELECT {', '.join(select)} FROM {SOURCE_PLAYERS} p {joins};"
        "CREATE VIEW players AS SELECT * FROM all_players WHERE status NOT IN ('u','n');")


def open_read_only(db: bytes) -> sqlite3.Connection:
    """Load the snapshot into memory, add the model's tables, then only ever allow reads on it."""
    con = sqlite3.connect(":memory:")
    con.deserialize(db)
    _add_model_tables(con)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
    return con


def run_sql(con: sqlite3.Connection, sql: str) -> tuple[list[int], list, str | None]:
    """Returns (player ids, stats, stat label). The stat is the second column, if there is one."""
    if not is_select(sql):
        raise ValueError("Only SELECT queries are allowed")
    # Only the first MAX_ROWS rows are read anyway, and a stray "limit 0" would hide every answer.
    sql = re.sub(r"\s+LIMIT\s+\d+(\s*,\s*\d+|\s+OFFSET\s+\d+)?\s*$", "", sql, flags=re.IGNORECASE)
    ticks = 0

    def budget():
        nonlocal ticks
        ticks += 1
        return int(ticks * _OPS_PER_TICK > MAX_SQLITE_OPS)  # non-zero aborts the query

    con.set_progress_handler(budget, _OPS_PER_TICK)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        columns = [d[0] for d in cur.description or []]
    finally:
        con.set_progress_handler(None, 0)
    ids = [int(r[0]) for r in rows]
    if len(columns) < 2:
        return ids, [None] * len(rows), None
    label = stat_label(columns[1])

    def shown(value):
        if isinstance(value, float):
            return round(value, 2)
        # Text only means something with a label (a date of birth, not a bare status code).
        return value if isinstance(value, int) or label else None

    return ids, [shown(r[1]) for r in rows], label
