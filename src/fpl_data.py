"""Build the FPL SQLite database from the FPL API's JSON.

Returns the database as bytes so it can be stored in KV and loaded into memory by each Worker
instance.
"""

import sqlite3
import unicodedata

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
PUBLIC_COLUMNS = [
    "id", "code", "web_name", "first_name", "second_name", "team", "team_code",
    "position", "price", "total_points",
]

SCHEMA = """
DROP VIEW IF EXISTS team_fixtures;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS teams;
DROP TABLE IF EXISTS fixtures;
DROP TABLE IF EXISTS gameweeks;

CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    code INTEGER,
    name TEXT,
    short_name TEXT,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER
);

CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    code INTEGER,
    web_name TEXT,
    first_name TEXT,
    second_name TEXT,
    name_search TEXT,
    team_id INTEGER REFERENCES teams(id),
    team TEXT,
    team_name TEXT,
    team_code INTEGER,
    position TEXT,
    price REAL,
    total_points INTEGER,
    event_points INTEGER,
    form REAL,
    points_per_game REAL,
    ep_next REAL,
    selected_by_percent REAL,
    minutes INTEGER,
    starts INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    saves INTEGER,
    bonus INTEGER,
    bps INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    penalties_missed INTEGER,
    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL,
    expected_goals_per_90 REAL,
    expected_assists_per_90 REAL,
    expected_goal_involvements_per_90 REAL,
    defensive_contribution INTEGER,
    defensive_contribution_per_90 REAL,
    ict_index REAL,
    status TEXT,
    chance_of_playing_next_round INTEGER,
    news TEXT,
    transfers_in_event INTEGER,
    transfers_out_event INTEGER,
    cost_change_start REAL,
    penalties_order INTEGER,
    direct_freekicks_order INTEGER,
    corners_and_indirect_freekicks_order INTEGER,
    birth_date TEXT
);

CREATE TABLE fixtures (
    id INTEGER PRIMARY KEY,
    gameweek INTEGER,
    kickoff_time TEXT,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    home_difficulty INTEGER,
    away_difficulty INTEGER,
    finished INTEGER
);

CREATE TABLE gameweeks (
    id INTEGER PRIMARY KEY,
    deadline_time TEXT,
    is_current INTEGER,
    is_next INTEGER,
    finished INTEGER
);

-- One row per team per fixture, from that team's point of view.
CREATE VIEW team_fixtures AS
SELECT f.id AS fixture_id, f.gameweek, f.kickoff_time, f.home_team_id AS team_id,
       f.away_team_id AS opponent_id, 1 AS is_home, f.home_difficulty AS difficulty, f.finished
FROM fixtures f
UNION ALL
SELECT f.id, f.gameweek, f.kickoff_time, f.away_team_id, f.home_team_id, 0, f.away_difficulty, f.finished
FROM fixtures f;
"""


def normalise(text: str) -> str:
    """Lowercase and strip accents so 'odegaard' matches 'Ødegaard'."""
    text = text.replace("ø", "o").replace("Ø", "O")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()



def _num(value):
    return None if value is None or value == "" else float(value)



def build_db(bootstrap: dict, fixtures: list) -> bytes:
    teams = {t["id"]: t for t in bootstrap["teams"]}
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)

    con.executemany(
        "INSERT INTO teams VALUES (?, ?, ?, ?, ?, ?)",
        [
            (t["id"], t["code"], t["name"], t["short_name"],
             t["strength_overall_home"], t["strength_overall_away"])
            for t in bootstrap["teams"]
        ],
    )

    rows = []
    for p in bootstrap["elements"]:
        t = teams[p["team"]]
        rows.append({
            "id": p["id"],
            "code": p["code"],
            "web_name": p["web_name"],
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "name_search": normalise(f"{p['first_name']} {p['second_name']} {p['web_name']}"),
            "team_id": t["id"],
            "team": t["short_name"],
            "team_name": t["name"],
            "team_code": t["code"],
            "position": POSITIONS[p["element_type"]],
            "price": p["now_cost"] / 10,
            "total_points": p["total_points"],
            "event_points": p["event_points"],
            "form": _num(p["form"]),
            "points_per_game": _num(p["points_per_game"]),
            "ep_next": _num(p["ep_next"]),
            "selected_by_percent": _num(p["selected_by_percent"]),
            "minutes": p["minutes"],
            "starts": p["starts"],
            "goals_scored": p["goals_scored"],
            "assists": p["assists"],
            "clean_sheets": p["clean_sheets"],
            "goals_conceded": p["goals_conceded"],
            "saves": p["saves"],
            "bonus": p["bonus"],
            "bps": p["bps"],
            "yellow_cards": p["yellow_cards"],
            "red_cards": p["red_cards"],
            "penalties_missed": p["penalties_missed"],
            "expected_goals": _num(p["expected_goals"]),
            "expected_assists": _num(p["expected_assists"]),
            "expected_goal_involvements": _num(p["expected_goal_involvements"]),
            "expected_goals_conceded": _num(p["expected_goals_conceded"]),
            "expected_goals_per_90": _num(p["expected_goals_per_90"]),
            "expected_assists_per_90": _num(p["expected_assists_per_90"]),
            "expected_goal_involvements_per_90": _num(p["expected_goal_involvements_per_90"]),
            "defensive_contribution": p["defensive_contribution"],
            "defensive_contribution_per_90": _num(p["defensive_contribution_per_90"]),
            "ict_index": _num(p["ict_index"]),
            "status": p["status"],
            "chance_of_playing_next_round": p["chance_of_playing_next_round"],
            "news": p["news"],
            "transfers_in_event": p["transfers_in_event"],
            "transfers_out_event": p["transfers_out_event"],
            "cost_change_start": p["cost_change_start"] / 10,
            "penalties_order": p["penalties_order"],
            "direct_freekicks_order": p["direct_freekicks_order"],
            "corners_and_indirect_freekicks_order": p["corners_and_indirect_freekicks_order"],
            "birth_date": p["birth_date"],
        })
    cols = list(rows[0])
    con.executemany(
        f"INSERT INTO players ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
        rows,
    )

    con.executemany(
        "INSERT INTO fixtures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (f["id"], f["event"], f["kickoff_time"], f["team_h"], f["team_a"],
             f["team_h_score"], f["team_a_score"], f["team_h_difficulty"],
             f["team_a_difficulty"], int(f["finished"]))
            for f in fixtures
        ],
    )

    con.executemany(
        "INSERT INTO gameweeks VALUES (?, ?, ?, ?, ?)",
        [
            (e["id"], e["deadline_time"], int(e["is_current"]), int(e["is_next"]), int(e["finished"]))
            for e in bootstrap["events"]
        ],
    )
    con.commit()
    data = con.serialize()
    con.close()
    return data
