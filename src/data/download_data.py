"""
Download and slice the transfermarkt-datasets DuckDB database
down to just the 5 leagues we care about, and export the tables we
need as CSVs into data/raw/.

Run from the project root:
    python src/data/download_data.py
"""
import duckdb
import os

DB_PATH = "data/raw/transfermarkt-datasets.duckdb"
OUT_DIR = "data/raw"

# Transfermarkt's competition IDs for the "big 5" domestic top flights.
# We print the matched competitions before exporting so you can eyeball
# that these are actually the right ones before trusting the filter.
BIG_5_COMPETITION_IDS = ["GB1", "ES1", "L1", "IT1", "FR1"]

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"Expected the database at {DB_PATH}. Download it first with:\n"
        f"  curl -L -o {DB_PATH} https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
    )

con = duckdb.connect(DB_PATH)

print("Tables in the database:")
print(con.execute("SHOW TABLES").fetchdf())

# Sanity check: which competitions do these IDs actually map to?
comps = con.execute(f"""
    SELECT competition_id, name, country_name, type
    FROM competitions
    WHERE competition_id IN ({','.join([f"'{c}'" for c in BIG_5_COMPETITION_IDS])})
""").fetchdf()
print("\nMatched competitions (verify these are the right 5 leagues):")
print(comps)

id_list_sql = ",".join([f"'{c}'" for c in BIG_5_COMPETITION_IDS])

# Players who are/were attached to a club in these leagues
players = con.execute(f"""
    SELECT p.*
    FROM players p
    JOIN clubs c ON p.current_club_id = c.club_id
    WHERE c.domestic_competition_id IN ({id_list_sql})
""").fetchdf()

# Appearances (per-game stats) only for games in these competitions
appearances = con.execute(f"""
    SELECT a.*
    FROM appearances a
    JOIN games g ON a.game_id = g.game_id
    WHERE g.competition_id IN ({id_list_sql})
""").fetchdf()

# Market valuations for the players we kept
valuations = con.execute("""
    SELECT v.*
    FROM player_valuations v
    JOIN (SELECT DISTINCT player_id FROM players
          WHERE current_club_id IN (SELECT club_id FROM clubs WHERE domestic_competition_id IN ({id_list_sql}))) p
    ON v.player_id = p.player_id
""".format(id_list_sql=id_list_sql)).fetchdf()

clubs = con.execute(f"""
    SELECT * FROM clubs WHERE domestic_competition_id IN ({id_list_sql})
""").fetchdf()

print(f"\nplayers: {len(players):,} rows")
print(f"appearances: {len(appearances):,} rows")
print(f"valuations: {len(valuations):,} rows")
print(f"clubs: {len(clubs):,} rows")

players.to_csv(f"{OUT_DIR}/players.csv", index=False)
appearances.to_csv(f"{OUT_DIR}/appearances.csv", index=False)
valuations.to_csv(f"{OUT_DIR}/player_valuations.csv", index=False)
comps.to_csv(f"{OUT_DIR}/competitions.csv", index=False)
clubs.to_csv(f"{OUT_DIR}/clubs.csv", index=False)

print(f"\nDone. CSVs written to {OUT_DIR}/")
con.close()