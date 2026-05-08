import sys, io
sys.path.insert(0, r"C:/Users/shank.subramani_betf/Desktop/ShotsDashboard/WNBA_RAPM/.claude/worktrees/youthful-grothendieck-4e6edc/wnba_origination")
import pandas as pd, numpy as np
import player_store, roster as roster_mod
from collections import defaultdict

TEAM_MAP = {
    "Portland Fire": "POR", "Connecticut Sun": "CON", "Seattle Storm": "SEA",
    "Toronto Tempo": "TOR", "Washington Mystics": "WAS", "Chicago Sky": "CHI",
    "Los Angeles Sparks": "LAS", "Dallas Wings": "DAL",
    "Golden State Valkyries": "GSV", "Minnesota Lynx": "MIN",
    "Indiana Fever": "IND", "Phoenix Mercury": "PHX", "Atlanta Dream": "ATL",
    "New York Liberty": "NYL", "Las Vegas Aces": "LVA",
}
STARTER_MINS = 28.0
BACKUP_MINS  = 12.0

RAW = """Player,Team,Position,Depth
C. Leite,Portland Fire,PG,Starter
S. Sutton,Portland Fire,PG,Backup
S. Barker,Portland Fire,SG,Starter
K. Smalls,Portland Fire,SG,Backup
B. Carleton,Portland Fire,SF,Starter
M. Caldwell,Portland Fire,SF,Backup
N. Puoch,Portland Fire,PF,Starter
H. Jones,Portland Fire,PF,Backup
M. Gustafson,Portland Fire,C,Starter
L. Geiselsoder,Portland Fire,C,Backup
L. Lacan,Connecticut Sun,PG,Starter
C. Leger-Walker,Connecticut Sun,PG,Backup
S. Rivers,Connecticut Sun,SG,Starter
N. Angloma,Connecticut Sun,SG,Backup
K. Burke,Connecticut Sun,SF,Starter
D. Miller,Connecticut Sun,SF,Backup
A. Edwards,Connecticut Sun,PF,Starter
A. Morrow,Connecticut Sun,PF,Backup
B. Griner,Connecticut Sun,C,Starter
O. Nelson-Ododa,Connecticut Sun,C,Backup
N. Hiedeman,Seattle Storm,PG,Starter
J. Melbourne,Seattle Storm,PG,Backup
F. Johnson,Seattle Storm,SG,Starter
L. Brown,Seattle Storm,SG,Backup
J. Horston,Seattle Storm,SF,Starter
K. Samuelson,Seattle Storm,SF,Backup
E. Magbegor,Seattle Storm,PF,Starter
A. Fam,Seattle Storm,PF,Backup
D. Malonga,Seattle Storm,C,Starter
S. Dolson,Seattle Storm,C,Backup
J. Alleman,Toronto Tempo,PG,Starter
K. Rice,Toronto Tempo,PG,Backup
B. Sykes,Toronto Tempo,SG,Starter
K. Nurse,Toronto Tempo,SG,Backup
M. Mabrey,Toronto Tempo,SF,Starter
A. Nye,Toronto Tempo,SF,Backup
L. Juskaite,Toronto Tempo,PF,Starter
I. Harrison,Toronto Tempo,PF,Backup
T. Fagbenle,Toronto Tempo,C,Starter
N. Sabally,Toronto Tempo,C,Backup
G. Amoore,Washington Mystics,PG,Starter
A. Wilson,Washington Mystics,PG,Backup
S. Citron,Washington Mystics,SG,Starter
L. Olsen,Washington Mystics,SG,Backup
M. Onyewere,Washington Mystics,SF,Starter
C. McMahon,Washington Mystics,SF,Backup
K. Iriafen,Washington Mystics,PF,Starter
A. Dugalic,Washington Mystics,PF,Backup
S. Austin,Washington Mystics,C,Starter
L. Betts,Washington Mystics,C,Backup
S. Diggins,Chicago Sky,PG,Starter
N. Cloud,Chicago Sky,PG,Backup
J. Sheldon,Chicago Sky,SG,Starter
R. Banham,Chicago Sky,SG,Backup
R. Jackson,Chicago Sky,SF,Starter
M. Jaquez,Chicago Sky,SF,Backup
A. Stevens,Chicago Sky,PF,Starter
D. Carrington,Chicago Sky,PF,Backup
K. Cardoso,Chicago Sky,C,Starter
E. Williams,Chicago Sky,C,Backup
E. Wheeler,Los Angeles Sparks,PG,Starter
J. Vanloo,Los Angeles Sparks,PG,Backup
K. Plum,Los Angeles Sparks,SG,Starter
T. Latson,Los Angeles Sparks,SG,Backup
A. Atkins,Los Angeles Sparks,SF,Starter
R. Burrell,Los Angeles Sparks,SF,Backup
N. Ogwumike,Los Angeles Sparks,PF,Starter
E. Cannon,Los Angeles Sparks,PF,Backup
D. Hamby,Los Angeles Sparks,C,Starter
C. Brink,Los Angeles Sparks,C,Backup
P. Bueckers,Dallas Wings,PG,Starter
O. Sims,Dallas Wings,PG,Backup
A. Ogunbowale,Dallas Wings,SG,Starter
A. James,Dallas Wings,SG,Backup
A. Fudd,Dallas Wings,SF,Starter
A. Clark,Dallas Wings,SF,Backup
J. Shepard,Dallas Wings,PF,Starter
M. Siegrist,Dallas Wings,PF,Backup
A. Smith,Dallas Wings,C,Starter
L. Yueru,Dallas Wings,C,Backup
V. Burton,Golden State Valkyries,PG,Starter
K. Chen,Golden State Valkyries,PG,Backup
T. Hayes,Golden State Valkyries,SG,Starter
J. Joctel,Golden State Valkyries,SG,Backup
G. Williams,Golden State Valkyries,SF,Starter
V. Vandalsini,Golden State Valkyries,SF,Backup
S. Thornton,Golden State Valkyries,PF,Starter
J. Salaun,Golden State Valkyries,PF,Backup
K. Stokes,Golden State Valkyries,C,Starter
I. Rupert,Golden State Valkyries,C,Backup
O. Miles,Minnesota Lynx,PG,Starter
J. Sherrod,Minnesota Lynx,PG,Backup
C. Williams,Minnesota Lynx,SG,Starter
A. Delaere,Minnesota Lynx,SG,Backup
K. McBride,Minnesota Lynx,SF,Starter
N. Coffey,Minnesota Lynx,SF,Backup
N. Collier,Minnesota Lynx,PF,Starter
N. Howard,Minnesota Lynx,PF,Backup
D. Juhasz,Minnesota Lynx,C,Starter
E. Cechova,Minnesota Lynx,C,Backup
C. Clark,Indiana Fever,PG,Starter
R. Johnson,Indiana Fever,PG,Backup
K. Mitchell,Indiana Fever,SG,Starter
S. Walker-Kimbrough,Indiana Fever,SG,Backup
S. Cunningham,Indiana Fever,SF,Starter
L. Hull,Indiana Fever,SF,Backup
M. Billings,Indiana Fever,PF,Starter
D. Dantas,Indiana Fever,PF,Backup
A. Boston,Indiana Fever,C,Starter
M. Timpson,Indiana Fever,C,Backup
M. Makanai,Phoenix Mercury,PG,Starter
K. Williams,Phoenix Mercury,PG,Backup
S. Whitcomb,Phoenix Mercury,SG,Starter
N. Brochant,Phoenix Mercury,SG,Backup
K. Copper,Phoenix Mercury,SF,Starter
V. Ayayi,Phoenix Mercury,SF,Backup
A. Thomas,Phoenix Mercury,PF,Starter
D. Bonner,Phoenix Mercury,PF,Backup
N. Mack,Phoenix Mercury,C,Starter
K. Linskens,Phoenix Mercury,C,Backup
J. Canada,Atlanta Dream,PG,Starter
M. Cazorla,Atlanta Dream,PG,Backup
A. Gray,Atlanta Dream,SG,Starter
T. Paopao,Atlanta Dream,SG,Backup
R. Howard,Atlanta Dream,SF,Starter
M. Borlase,Atlanta Dream,SF,Backup
A. Reese,Atlanta Dream,PF,Starter
N. Hillmon,Atlanta Dream,PF,Backup
B. Jones,Atlanta Dream,C,Starter
M. Okot,Atlanta Dream,C,Backup
S. Ionescu,New York Liberty,PG,Starter
P. Astier,New York Liberty,PG,Backup
B. Laney-Hamilton,New York Liberty,SG,Starter
M. Johannes,New York Liberty,SG,Backup
L. Fiebich,New York Liberty,SF,Starter
R. Allen,New York Liberty,SF,Backup
B. Stewart,New York Liberty,PF,Starter
S. Sabally,New York Liberty,PF,Backup
J. Jones,New York Liberty,C,Starter
H. Xu,New York Liberty,C,Backup
C. Gray,Las Vegas Aces,PG,Starter
D. Evans,Las Vegas Aces,PG,Backup
J. Young,Las Vegas Aces,SG,Starter
C. Carter,Las Vegas Aces,SG,Backup
K. Bell,Las Vegas Aces,SF,Starter
J. Loyd,Las Vegas Aces,SF,Backup
N. Smith,Las Vegas Aces,PF,Starter
S. Talbot,Las Vegas Aces,PF,Backup
A. Wilson,Las Vegas Aces,C,Starter
C. Parker-Tyus,Las Vegas Aces,C,Backup"""

store = player_store.load()

# Build last-name lookup: last_name -> [(player_id, full_name, team_abbr)]
last_lookup = defaultdict(list)
for _, r in store.iterrows():
    parts = str(r["player_name"]).split()
    if parts:
        last_lookup[parts[-1].lower()].append((int(r["player_id"]), r["player_name"], r["team_abbr"]))

def match_player(name_abbr, team_abbr):
    raw = name_abbr.strip()
    dot = raw.find(".")
    if dot == -1:
        return np.nan, raw
    first_init = raw[:dot].strip().upper()
    rest = raw[dot+1:].strip()
    # last word of rest handles hyphenated names — try full and suffix
    last_tokens = [rest.split()[-1].lower(), rest.lower().replace("-", "").replace(" ", "")]
    # also try each part of hyphenated
    last_tokens += [p.lower() for p in rest.split()[-1].split("-")]

    for last in dict.fromkeys(last_tokens):  # deduplicate, preserve order
        candidates = last_lookup.get(last, [])
        matches = [(pid, fn, ta) for pid, fn, ta in candidates
                   if fn.split()[0][0].upper() == first_init]
        if not matches:
            continue
        # prefer same team
        same = [(pid, fn) for pid, fn, ta in matches if ta == team_abbr]
        if same:
            return same[0]
        return matches[0][0], matches[0][1]
    return np.nan, raw

df = pd.read_csv(io.StringIO(RAW))
df["minutes"] = df["Depth"].map({"Starter": STARTER_MINS, "Backup": BACKUP_MINS})
df["abbr"] = df["Team"].map(TEAM_MAP)

rows_out, unmatched = [], []
for _, row in df.iterrows():
    pid, full_name = match_player(row["Player"], row["abbr"])
    rows_out.append({"player_id": pid, "player_name": full_name,
                     "projected_minutes": row["minutes"], "team_abbr": row["abbr"]})
    if pd.isna(pid):
        unmatched.append(f"  {row['Player']} ({row['abbr']})")

result_df = pd.DataFrame(rows_out)
matched = result_df["player_id"].notna().sum()
print(f"Matched {matched}/{len(result_df)} players to store")

if unmatched:
    print(f"\nUnmatched ({len(unmatched)}) — saved by name only, RAPM will be 0:")
    for u in unmatched:
        print(u)

for abbr, group in result_df.groupby("team_abbr"):
    rot = group[["player_id", "player_name", "projected_minutes"]].copy()
    roster_mod.set_rotation(abbr, rot)

print("\nSaved rotations for:", sorted(result_df["team_abbr"].unique()))
