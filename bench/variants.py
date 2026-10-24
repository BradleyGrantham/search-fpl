"""Prompt and schema variants of the /ask pipeline, for A/B runs with bench/run.py.

A variant can change the database the SQL runs on (`setup`, applied to the snapshot before it
is locked read-only), the system prompt, how the result is read back, and extra API
parameters. `worker` is exactly what the Worker does now (src/legacy.py, src/llm.py); `baseline` is
what it did before the October 2026 latency work (legacy.py), and the experiments build on that.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

import legacy
import llm
import nl

# --- running SQL --------------------------------------------------------------


def open_db(db: bytes, setup: Callable | None) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.deserialize(db)
    if setup:
        setup(con)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
    return con


def run_baseline(con, sql, labels):
    ids, stats = legacy.run_sql(con, sql)
    return ids, stats, legacy.stat_label(sql)


def run_positional(con, sql, labels, strip_limit=False):
    """The stat is whatever the second column is; rounding and the row cap happen here."""
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    if strip_limit:  # we only read 10 rows anyway; a stray "limit 0" would hide every answer
        sql = re.sub(r"\s+LIMIT\s+\d+(\s*,\s*\d+|\s+OFFSET\s+\d+)?\s*$", "", sql, flags=re.IGNORECASE)
    ticks = 0

    def budget():
        nonlocal ticks
        ticks += 1
        return int(ticks * legacy._OPS_PER_TICK > legacy.MAX_SQLITE_OPS)

    con.set_progress_handler(budget, legacy._OPS_PER_TICK)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(legacy.MAX_ROWS)
        desc = cur.description or []
    finally:
        con.set_progress_handler(None, 0)
    has_stat = len(desc) > 1
    stats = [(round(r[1], 2) if isinstance(r[1], float) else r[1]) if has_stat else None for r in rows]
    return [int(r[0]) for r in rows], stats, label_for(desc[1][0], labels) if has_stat else None


def label_for(column: str, labels: dict) -> str | None:
    known = {c for c in re.findall(r"[a-z_][a-z0-9_]*", column.lower()) if c in labels}
    return labels[known.pop()] if len(known) == 1 else None


# --- schema changes -------------------------------------------------------------

# Long names -> short ones (E4). Everything not listed keeps its name.
SHORT = {
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
PER_90 = ["expected_goals_per_90", "expected_assists_per_90", "expected_goal_involvements_per_90",
          "defensive_contribution_per_90"]


def players_setup(derived: bool = False, short: bool = False):
    """`players` becomes a view of available players over `all_players`, optionally with
    precomputed columns (E3) and short names (E4)."""

    def setup(con: sqlite3.Connection):
        cols = [r[1] for r in con.execute("PRAGMA table_info(players)")]
        name = (lambda c: SHORT.get(c, c)) if short else (lambda c: c)
        select = []
        for c in cols:
            if derived and c in PER_90:  # per-90 on a tiny sample is noise: blank it
                select.append(f"CASE WHEN p.minutes >= 270 THEN p.{c} END AS {name(c)}")
            else:
                select.append(f"p.{c} AS {name(c)}")
        joins = ""
        if derived:
            (gw,) = con.execute("SELECT COALESCE((SELECT id FROM gameweeks WHERE is_next = 1), 38)").fetchone()
            select += [
                "p.minutes >= 90 AS regular",
                "ROUND(p.total_points / p.price, 2) AS pts_per_m",
                "ROUND(p.goals_scored - p.expected_goals, 2) AS xg_diff",
                "p.price <= CASE p.position WHEN 'MID' THEN 5.5 WHEN 'FWD' THEN 6.0 ELSE 4.5 END AS budget",
                "COALESCE(f1.n, 0) AS fixtures_next", "f1.home AS home_next",
                "f1.fdr AS fdr_next", "f3.fdr AS fdr_next3", "f5.fdr AS fdr_next5",
            ]
            agg = ("(SELECT team_id, COUNT(*) AS n, MAX(is_home) AS home, ROUND(AVG(difficulty), 2) AS fdr "
                   "FROM team_fixtures WHERE gameweek BETWEEN {a} AND {b} GROUP BY team_id)")
            joins = " ".join(
                f"LEFT JOIN {agg.format(a=gw, b=gw + k - 1)} {alias} ON {alias}.team_id = p.team_id"
                for alias, k in (("f1", 1), ("f3", 3), ("f5", 5)))
        con.executescript(
            f"CREATE TABLE all_players AS SELECT {', '.join(select)} FROM players p {joins};"
            "DROP TABLE players;"
            "CREATE VIEW players AS SELECT * FROM all_players WHERE status NOT IN ('u','n');")

    return setup


def short_labels() -> dict:
    labels = {SHORT.get(k, k): v for k, v in legacy.STAT_LABELS.items()}
    labels.update({"pts_per_m": "pts/£m", "xg_diff": "G−xG", "fdr_next": "FDR", "fdr_next3": "FDR",
                   "fdr_next5": "FDR", "fixtures_next": "fixtures"})
    return labels


def long_labels() -> dict:
    labels = dict(legacy.STAT_LABELS)
    labels.update({"pts_per_m": "pts/£m", "xg_diff": "G−xG", "fdr_next": "FDR", "fdr_next3": "FDR",
                   "fdr_next5": "FDR", "fixtures_next": "fixtures"})
    return labels


# --- prompts ---------------------------------------------------------------------

def _gw_and_teams(con):
    teams = con.execute("SELECT short_name, name FROM teams ORDER BY name").fetchall()
    row = con.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
    return (row[0] if row else 38), ", ".join(f"{s}={n}" for s, n in teams)


def prompt_padded(con):
    """Baseline plus a repeated copy of itself: same answers, ~2x the cached input."""
    base = legacy.build_system_prompt(con)
    return base + "\n\nReference copy of the instructions above (identical, for lookup only):\n" + base


def prompt_codecuts(con):
    """E1: no LIMIT, no ROUND, no 'AS stat' - the code does those."""
    gw, teams = _gw_and_teams(con)
    return f"""\
Write ONE SQLite SELECT statement that answers the user's question about Fantasy Premier League players.
Output only the SQL. No markdown, no explanation.

The result MUST have columns:
  1. id  (players.id)
  2. (optional) the single number most relevant to the question (e.g. price for 'cheap', expected_goals for 'xG').
     Omit it if nothing specific was asked about. No alias or rounding needed.
Order rows so the best answers come first. Only the first 10 rows are used, so no LIMIT.
Unless the question is about injuries/unavailable players, exclude players with status 'u' or 'n'.
For recommendation-style questions (in form, differentials, easy fixtures, value, 'who should I buy'), only include
players who actually play (minutes >= 90 so far) and rank by the quality asked for, not by price.
'Cheap'/'budget' with no number is a price cap (GKP <= 4.5, DEF <= 4.5, MID <= 5.5, FWD <= 6.0), not a sort order.
'Premium' means price >= 9.5.
Per-90 stats are meaningless on tiny samples: whenever you filter or rank by a *_per_90 column, require minutes >= 270.
If the question is just a player's name, match it with name_search.
If the question cannot be answered from this data, output: SELECT id FROM players WHERE 0

The next gameweek is {gw}; the current one is {gw - 1}.
Teams (short_name=name): {teams}

Schema:
{legacy.DATA_DICTIONARY}
Examples:
Q: arsenal defenders under 5m
SELECT id, price FROM players WHERE team = 'ARS' AND position = 'DEF' AND price <= 5.0 AND status NOT IN ('u','n') ORDER BY total_points DESC

Q: midfielders with the best xG per 90 who've played at least 300 mins
SELECT id, expected_goals_per_90 FROM players WHERE position = 'MID' AND minutes >= 300 AND status NOT IN ('u','n') ORDER BY 2 DESC

Q: cheap in-form midfielders
SELECT id, form FROM players WHERE position = 'MID' AND price <= 5.5 AND minutes >= 90 AND status NOT IN ('u','n') ORDER BY 2 DESC

Q: forwards with easy fixtures over the next 3 gameweeks
SELECT p.id, AVG(tf.difficulty) FROM players p JOIN team_fixtures tf ON tf.team_id = p.team_id WHERE p.position = 'FWD' AND p.minutes >= 90 AND p.status NOT IN ('u','n') AND tf.gameweek BETWEEN {gw} AND {gw} + 2 GROUP BY p.id ORDER BY 2, p.total_points DESC
"""


def prompt_active(con):
    """E2: E1 + `players` holds only available players; `all_players` has everyone."""
    p = prompt_codecuts(con)
    p = p.replace(
        "Unless the question is about injuries/unavailable players, exclude players with status 'u' or 'n'.\n",
        "`players` already leaves out players who left or aren't eligible (status 'u'/'n'); query `all_players` "
        "(same columns) only when asked about those.\n")
    p = p.replace("players: one row per FPL player, current season to date.",
                  "players: one row per available FPL player, current season to date (all_players: everyone).")
    p = p.replace(" AND status NOT IN ('u','n')", "").replace(" AND p.status NOT IN ('u','n')", "")
    p = p.replace("p.position = 'FWD' AND p.minutes >= 90 AND p.status NOT IN ('u','n')", "p.position = 'FWD' AND p.minutes >= 90")
    return p


DERIVED_DICT = """\
  regular: 1 if minutes >= 90 (actually plays)
  budget: 1 if cheap for the position (GKP/DEF <= 4.5, MID <= 5.5, FWD <= 6.0)
  pts_per_m: total_points / price (value); xg_diff: goals_scored - expected_goals (over/underperformance)
  fixtures_next: fixtures next gameweek (2 = double, 0 = blank); home_next: 1 if at home next gameweek
  fdr_next, fdr_next3, fdr_next5: average fixture difficulty over the next 1/3/5 gameweeks (1 easy - 5 hard)
  *_per_90 columns are NULL under 270 minutes
"""


def prompt_derived(con):
    """E3: E2 + precomputed columns replace the fixture join, per-90 rule, budget rule, etc."""
    gw, teams = _gw_and_teams(con)
    dictionary = legacy.DATA_DICTIONARY.replace(
        "  birth_date: 'YYYY-MM-DD'\n", "  birth_date: 'YYYY-MM-DD'\n" + DERIVED_DICT)
    dictionary = dictionary.replace("players: one row per FPL player, current season to date.",
                                    "players: one row per available FPL player, current season to date (all_players: everyone).")
    return f"""\
Write ONE SQLite SELECT statement that answers the user's question about Fantasy Premier League players.
Output only the SQL. No markdown, no explanation.

The result MUST have columns:
  1. id  (players.id)
  2. (optional) the single number most relevant to the question (e.g. price for 'cheap', expected_goals for 'xG').
     Omit it if nothing specific was asked about. No alias or rounding needed.
Order rows so the best answers come first. Only the first 10 rows are used, so no LIMIT.
`players` already leaves out players who left or aren't eligible (status 'u'/'n'); query `all_players` (same columns) only when asked about those.
For recommendation-style questions (in form, differentials, easy fixtures, value, 'who should I buy'), only include
regular players and rank by the quality asked for, not by price.
'Cheap'/'budget' with no number means budget = 1, not a sort order. 'Premium' means price >= 9.5.
If the question is just a player's name, match it with name_search.
If the question cannot be answered from this data, output: SELECT id FROM players WHERE 0

The next gameweek is {gw}; the current one is {gw - 1}.
Teams (short_name=name): {teams}

Schema:
{dictionary}
Examples:
Q: arsenal defenders under 5m
SELECT id, price FROM players WHERE team = 'ARS' AND position = 'DEF' AND price <= 5.0 ORDER BY total_points DESC

Q: midfielders with the best xG per 90
SELECT id, expected_goals_per_90 FROM players WHERE position = 'MID' ORDER BY 2 DESC NULLS LAST

Q: cheap in-form midfielders
SELECT id, form FROM players WHERE position = 'MID' AND budget AND regular ORDER BY 2 DESC

Q: forwards with easy fixtures over the next 3 gameweeks
SELECT id, fdr_next3 FROM players WHERE position = 'FWD' AND regular ORDER BY 2, total_points DESC
"""


def prompt_short(con):
    """E4: E3 with short column names."""
    p = prompt_derived(con)
    # Rename longest first so a name inside another isn't hit first.
    for old in sorted(SHORT, key=len, reverse=True):
        p = re.sub(rf"\b{old}\b", SHORT[old], p)
    p = p.replace("*_per_90 columns are NULL", "per-90 columns (xg90, xa90, xgi90, dc90) are NULL")
    return p


def prompt_terse(con):
    """E5: E4's content, rewritten as tightly as possible."""
    gw, teams = _gw_and_teams(con)
    return f"""\
Answer with ONE SQLite SELECT over the FPL tables below. Output SQL only.
Columns: id, then optionally the one number the question is about (e.g. price for "cheap"). Best first; first 10 rows are used (no LIMIT/ROUND/alias).
players = available players; all_players also has those who left (status u/n): use it only if asked about them.
Recommendations (form, differentials, fixtures, value, who to buy): regular only, rank by that quality, not price.
cheap/budget = budget; premium = price >= 9.5. A bare name: name_search LIKE '%name%'. Unanswerable: SELECT id FROM players WHERE 0
Next gameweek {gw}. Teams: {teams}

players: id, web_name, name_search (lowercase 'first last web_name'), team (code), team_name, position (GKP/DEF/MID/FWD),
price (£m), pts, gw_pts (this gameweek), form, ppg, xpts_next, own (%; differential < 10), minutes, starts, goals, assists,
clean_sheets, conceded, saves, bonus, bps, yellow_cards, red_cards, pens_missed, xg, xa, xgi, xgc, xg90, xa90, xgi90, dc, dc90
(per-90s NULL under 270 mins), ict_index, status (a ok, d doubtful, i injured, s suspended), chance (0-100, NULL = fit), news,
t_in, t_out (this gameweek), price_change (since start), pen_order, fk_order, corner_order (1 = first choice, NULL = none),
birth_date, regular (minutes >= 90), budget (cheap for position), pts_per_m, xg_diff (goals - xg), fixtures_next (2 double, 0 blank),
home_next, fdr_next, fdr_next3, fdr_next5 (avg difficulty next 1/3/5 GWs, 1 easy - 5 hard)
team_fixtures: team_id, gameweek, opponent_id, is_home, difficulty. teams: id, name, short_name

Q: arsenal defenders under 5m
SELECT id, price FROM players WHERE team = 'ARS' AND position = 'DEF' AND price <= 5 ORDER BY pts DESC
Q: cheap in-form midfielders
SELECT id, form FROM players WHERE position = 'MID' AND budget AND regular ORDER BY 2 DESC
Q: forwards with easy fixtures next 3 gameweeks
SELECT id, fdr_next3 FROM players WHERE position = 'FWD' AND regular ORDER BY 2, pts DESC
"""


def prompt_lean(con):
    """E6: terse + a fix for each regression found in E1-E5, and SQL-only replies."""
    gw, teams = _gw_and_teams(con)
    return f"""\
Turn the user's message into ONE SQLite SELECT over the FPL tables below. Reply with SQL only, never prose:
if the message is unclear, off-topic or not about FPL players, reply SELECT id FROM players WHERE 0
Columns: id, then optionally the one number the question is about (e.g. price for "cheap"). Best first; first 10 rows are used (no LIMIT/ROUND/alias).
players = available players; all_players also has those who left (status u/n). A name (even misspelt): all_players WHERE name_search LIKE '%name%'.
Recommendations (form, differentials, fixtures, value, who to buy): add regular and rank by that quality, not price. Plain lists and filters don't need regular.
under/over Xm = price <= X / >= X. cheap/budget = budget; premium = price >= 9.5; template = most owned.
Next gameweek {gw}. Teams: {teams}

players: id, web_name, name_search (lowercase 'first last web_name'), team (code), team_name, position (GKP/DEF/MID/FWD),
price (£m), pts, gw_pts (this gameweek), form, ppg, xpts_next, own (%; differential < 10), minutes, starts, goals, assists,
clean_sheets, conceded, saves, bonus, bps, yellow_cards, red_cards, pens_missed, xg, xa, xgi, xgc, xg90, xa90, xgi90, dc, dc90
(per-90s NULL under 270 mins), ict_index, status (a ok, d doubtful, i injured, s suspended), chance (0-100, NULL = fit), news,
t_in, t_out (this gameweek), price_change (since start), pen_order, fk_order, corner_order (1 = first choice, NULL = none),
birth_date, regular (minutes >= 90), budget (cheap for position), pts_per_m, xg_diff (goals - xg), fixtures_next (2 double, 0 blank),
home_next, fdr_next, fdr_next3, fdr_next5 (avg difficulty next 1/3/5 GWs, 1 easy - 5 hard)
team_fixtures: team_id, gameweek, opponent_id, is_home, difficulty. teams: id, name, short_name

Q: arsenal defenders under 5m
SELECT id, price FROM players WHERE team = 'ARS' AND position = 'DEF' AND price <= 5 ORDER BY pts DESC
Q: cheap in-form midfielders
SELECT id, form FROM players WHERE position = 'MID' AND budget AND regular ORDER BY 2 DESC
Q: forwards with easy fixtures next 3 gameweeks
SELECT id, fdr_next3 FROM players WHERE position = 'FWD' AND regular ORDER BY 2, pts DESC
"""


def prompt_lean2(con):
    """E8: E6 with the 'unclear' rule narrowed, so vague FPL questions still get an answer."""
    return prompt_lean(con).replace(
        "Reply with SQL only, never prose:\n"
        "if the message is unclear, off-topic or not about FPL players, reply SELECT id FROM players WHERE 0\n",
        "Reply with SQL only, never prose.\n"
        "Vague FPL questions (who to buy, captain, any tips) are recommendations. "
        "Only if the message isn't about FPL at all: SELECT id FROM players WHERE 0\n")


def prompt_lean3(con):
    """E10: E8 + fixes for its four regressions (minute floor on 'lowest', plain lists by pts)."""
    return prompt_lean2(con).replace(
        "Plain lists and filters don't need regular.",
        "Plain lists and filters (a team's forwards, players under X, young players) don't need regular; rank them by pts.\n"
        "Lowest/fewest of a stat: minutes >= 270.")


def prompt_lean4(con):
    """E17: E10 + games played stated, and a misspelt-name example."""
    gw, _ = _gw_and_teams(con)
    p = prompt_lean3(con).replace(f"Next gameweek {gw}.", f"Next gameweek {gw} ({gw - 1} played).")
    return p + "Q: haalnd\nSELECT id FROM all_players WHERE name_search LIKE '%haaland%'\n"


def prompt_lower4(con):
    """E18: E17 as lowercase SQL with position flags."""
    gw, _ = _gw_and_teams(con)
    p = prompt_lower(con).replace(f"Next gameweek {gw}.", f"Next gameweek {gw} ({gw - 1} played).")
    return p + "Q: haalnd\nselect id from all_players where name_search like '%haaland%'\n"


def prompt_lower5(con):
    """E20: E18 with 'chance' 100 = fit, and games played stated as the max of starts."""
    gw, _ = _gw_and_teams(con)
    p = prompt_lower4(con)
    p = p.replace("chance (0-100, NULL = fit)", "chance (0-100; NULL or 100 = fit)")
    p = p.replace(f"Next gameweek {gw} ({gw - 1} played).",
                  f"Next gameweek {gw}; {gw - 1} played, so starts = {gw - 1} means every game.")
    return p


def prompt_lower7(con):
    """E22: E20 with non-player questions spelled out as unanswerable."""
    return prompt_lower5(con).replace(
        "Only if the message isn't about FPL at all: select id from players where 0",
        "If it isn't asking for players (the weather, who wins the league, a poem): select id from players where 0")


POSITIONS = {"GKP": "gkp", "DEF": "def", "MID": "mid", "FWD": "fwd"}


def position_flags(setup, aliases=False):
    """Adds gkp/def/mid/fwd 0/1 columns to all_players (and so to the players view)."""
    def wrapped(con):
        setup(con)
        flags = ", ".join(f"position = '{p}' AS {f}" for p, f in POSITIONS.items())
        if aliases:  # names the model sometimes reaches for; not in the prompt
            flags += ", position = 'GKP' AS gk"
        con.executescript(
            "DROP VIEW players; ALTER TABLE all_players RENAME TO _ap;"
            f"CREATE TABLE all_players AS SELECT *, {flags} FROM _ap; DROP TABLE _ap;"
            "CREATE VIEW players AS SELECT * FROM all_players WHERE status NOT IN ('u','n');")
    return wrapped


def _lower_sql(text):
    """Lowercase SQL keywords in example lines, and use position flags."""
    def fix(line):
        if not line.startswith("SELECT"):
            return line
        line = re.sub(r"position = '(\w+)'", lambda m: POSITIONS[m[1]], line)
        return re.sub(r"\b(SELECT|FROM|WHERE|AND|OR|ORDER BY|DESC|ASC|NOT|NULL|IS|IN|LIKE|GROUP BY)\b", lambda m: m[0].lower(), line)
    return "\n".join(fix(l) for l in text.split("\n"))


def prompt_lower(con):
    """E11: E10 written as lowercase SQL with position flags."""
    p = _lower_sql(prompt_lean3(con))
    p = p.replace("Turn the user's message into ONE SQLite SELECT", "Turn the user's message into ONE lowercase SQLite select")
    p = p.replace("reply SELECT id FROM players WHERE 0", "reply select id from players where 0")
    p = p.replace("Only if the message isn't about FPL at all: SELECT id FROM players WHERE 0",
                  "Only if the message isn't about FPL at all: select id from players where 0")
    p = p.replace("all_players WHERE name_search LIKE '%name%'", "all_players where name_search like '%name%'")
    p = p.replace("position (GKP/DEF/MID/FWD),", "position (GKP/DEF/MID/FWD; flags gkp, def, mid, fwd),")
    return p


def prompt_dsl(con):
    """E12: a one-line stat|where|order reply that the code turns into SQL."""
    gw, teams = _gw_and_teams(con)
    dictionary = prompt_lower(con).split("Next gameweek", 1)[1].split("\n", 1)[1].split("\nQ:", 1)[0]
    return f"""\
Turn the user's FPL question into one line: stat|where|order. Never prose.
We run: select id, <stat> from players where <where> order by <order>
stat: the one number the question is about (e.g. price for "cheap"), or blank. where: a lowercase SQLite condition, or blank.
order: blank means stat desc (pts desc if no stat). First 10 rows are shown.
For anything else (all_players, a join, group by) reply a full lowercase select whose first column is id.
players = available players; all_players also has those who left (status u/n). A name (even misspelt): select id from all_players where name_search like '%name%'
Recommendations (form, differentials, fixtures, value, who to buy, captain, tips): add regular and rank by that quality, not price.
Plain lists and filters (a team's forwards, players under X, young players) don't need regular; order them by pts desc.
Lowest/fewest of a stat: minutes >= 270. under/over Xm = price <= X / >= X. cheap/budget = budget; premium = price >= 9.5; template = most owned.
Not about FPL at all: |0
Next gameweek {gw}. Teams: {teams}
{dictionary}
Q: arsenal defenders under 5m
price|team = 'ARS' and def and price <= 5|pts desc
Q: cheap in-form midfielders
form|mid and budget and regular
Q: forwards with easy fixtures next 3 gameweeks
fdr_next3|fwd and regular|2, pts desc
Q: watkins
select id from all_players where name_search like '%watkins%'
"""


def prompt_mini(con):
    """E13: E10 without the team list or examples (cost only, if quality holds)."""
    p = prompt_lean3(con)
    p = re.sub(r"Next gameweek (\d+)\. Teams: .*\n", r"Next gameweek \1. team uses the usual 3-letter codes (TOT = Spurs, NFO = Forest).\n", p)
    return p.split("\nQ:", 1)[0] + "\n"


def is_sql(text):
    return bool(re.match(r"^\s*(SELECT|WITH)\b", text, re.IGNORECASE))


def is_dsl_answer(text):
    return is_sql(text) or "|" in text


def run_dsl(con, text, labels):
    if is_sql(text):
        return run_positional(con, text, labels)
    stat, where, order = (text.strip().split("|", 2) + ["", ""])[:3]
    stat, where, order = stat.strip(), where.strip(), order.strip()
    if order.lower() in ("asc", "desc"):  # a bare direction means "by the stat"
        order = f"2 {order}" if stat else f"pts {order}"
    sql = ("select id" + (f", {stat}" if stat else "") + " from players"
           + (f" where {where}" if where else "") + " order by " + (order or ("2 desc" if stat else "pts desc")))
    return run_positional(con, sql, labels)


# --- the variants ----------------------------------------------------------------

@dataclass
class Variant:
    name: str
    description: str
    prompt: Callable[[sqlite3.Connection], str]
    setup: Callable | None = None
    run: Callable = run_baseline
    labels: dict = field(default_factory=lambda: dict(legacy.STAT_LABELS))
    cache: bool = True
    api: dict = field(default_factory=dict)
    prose_is_empty: bool = False  # a reply that isn't SQL means "no answer": skip the retry
    is_answer: Callable = None  # what counts as "not prose" (default: starts with SELECT/WITH)
    hedge_ms: int | None = None  # send a second request if the first hasn't answered by then
    open: Callable | None = None  # opens the snapshot itself, instead of open_db(db, setup)
    via_llm: bool = False  # call llm.generate_sql itself (not with --stream, which needs its own call)


VARIANTS = {v.name: v for v in [
    Variant("worker", "What the Worker does now (E23: E22 + team columns, prose retried once)", nl.build_system_prompt,
            open=nl.open_read_only, run=lambda con, sql, labels: nl.run_sql(con, sql),
            via_llm=True,
            api={"max_tokens": llm.MAX_TOKENS, "stop_sequences": llm.STOP_SEQUENCES}),
    Variant("baseline", "What the Worker did before the latency work", legacy.build_system_prompt),
    Variant("baseline_b", "Identical copy of baseline (A/A noise check)", legacy.build_system_prompt),
    Variant("nocache", "Baseline without prompt caching", legacy.build_system_prompt, cache=False),
    Variant("padded", "Baseline with ~2x the cached input", prompt_padded),
    Variant("codecuts", "E1: no LIMIT/ROUND/AS stat in the SQL", prompt_codecuts, run=run_positional),
    Variant("active", "E2: E1 + players view of available players", prompt_active,
            setup=players_setup(), run=run_positional),
    Variant("derived", "E3: E2 + precomputed columns", prompt_derived,
            setup=players_setup(derived=True), run=run_positional, labels=long_labels()),
    Variant("short", "E4: E3 + short column names", prompt_short,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels()),
    Variant("terse", "E5: E4 with a rewritten, minimal prompt", prompt_terse,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels()),
    Variant("lean", "E6: E5 + regression fixes; prose replies return nothing", prompt_lean,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True),
    Variant("lean_low", "E7: E6 with effort low", prompt_lean,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, api={"output_config": {"effort": "low"}}),
    Variant("lean2", "E8: E6 with the 'unclear' rule narrowed", prompt_lean2,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True),
    Variant("lean2_low", "E9: E8 with effort low", prompt_lean2,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, api={"output_config": {"effort": "low"}}),
    Variant("lean3", "E10: E8 + fixes for its four regressions", prompt_lean3,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True),
    Variant("lower", "E11: E10 as lowercase SQL with position flags", prompt_lower,
            setup=position_flags(players_setup(derived=True, short=True)), run=run_positional,
            labels=short_labels(), prose_is_empty=True),
    Variant("dsl", "E12: one-line stat|where|order replies", prompt_dsl,
            setup=position_flags(players_setup(derived=True, short=True)), run=run_dsl,
            labels=short_labels(), prose_is_empty=True, is_answer=is_dsl_answer),
    Variant("stop", "E14: E8 + stop prose after its first sentence, max_tokens 200", prompt_lean2,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("geo_us", "E15: E8 served from the US only", prompt_lean2,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, api={"inference_geo": "us"}),
    Variant("hedge", "E16: E8 + a second request if no answer within 1 s", prompt_lean2,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, hedge_ms=1000),
    Variant("lean4", "E17: E10 + games played + misspelt-name example, stop on prose", prompt_lean4,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("lower4", "E18: E17 as lowercase SQL with position flags", prompt_lower4,
            setup=position_flags(players_setup(derived=True, short=True)), run=run_positional,
            labels=short_labels(), prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("lower5", "E20: E18 + 'chance 100 = fit' + games played spelled out", prompt_lower5,
            setup=position_flags(players_setup(derived=True, short=True)), run=run_positional,
            labels=short_labels(), prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("lower6", "E21: E20 + the code drops a trailing LIMIT and accepts gk for gkp", prompt_lower5,
            setup=position_flags(players_setup(derived=True, short=True), aliases=True),
            run=lambda con, sql, labels: run_positional(con, sql, labels, strip_limit=True),
            labels=short_labels(), prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("lower7", "E22: E21 + non-player questions spelled out as unanswerable", prompt_lower7,
            setup=position_flags(players_setup(derived=True, short=True), aliases=True),
            run=lambda con, sql, labels: run_positional(con, sql, labels, strip_limit=True),
            labels=short_labels(), prose_is_empty=True, api={"stop_sequences": [". ", ":\n"], "max_tokens": 200}),
    Variant("mini", "E13: E10 without team list or examples", prompt_mini,
            setup=players_setup(derived=True, short=True), run=run_positional, labels=short_labels(),
            prose_is_empty=True),
]}
