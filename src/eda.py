
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==================================
# 1. SETUP PROJECT PATH
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==================================
# 2. LOAD DATA
# ==================================

files = {
    "2024/25": "EPL_2024_2025.csv",
    "2025/26": "EPL_2025_2026.csv",
    "2026/27": "EPL_2026_2027.csv"
}

all_data = []

for season, filename in files.items():

    file_path = RAW_DIR / filename

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    df["Season"] = season

    all_data.append(df)

# Gop 3 mua giai
data = pd.concat(all_data, ignore_index=True)

print("Total rows:", len(data))

# ==================================
# 3. SELECT IMPORTANT COLUMNS
# ==================================

required_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "Season"
]

data = data[required_columns].copy()

# ==================================
# 4. DATA CLEANING
# ==================================

data["Date"] = pd.to_datetime(
    data["Date"],
    dayfirst=True,
    errors="coerce"
)

data["FTHG"] = pd.to_numeric(
    data["FTHG"], errors="coerce"
)

data["FTAG"] = pd.to_numeric(
    data["FTAG"], errors="coerce"
)

# Chi giu cac tran da co ket qua
data = data.dropna(
    subset=[
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR"
    ]
)

data["FTR"] = data["FTR"].str.strip().str.upper()

data = data[
    data["FTR"].isin(["H", "D", "A"])
].copy()

data["FTHG"] = data["FTHG"].astype(int)
data["FTAG"] = data["FTAG"].astype(int)

# Sap xep theo thoi gian
data = data.sort_values("Date").reset_index(drop=True)

# Luu dataset da xu ly
output_path = PROCESSED_DIR / "EPL_all_seasons.csv"

data.to_csv(output_path, index=False)

print("\nData loaded successfully!")
print("Completed matches:", len(data))
print("Saved to:", output_path)

print("\nMatches by season:")
print(data.groupby("Season").size())


# ==================================
# 5. MATCH RESULT DISTRIBUTION
# ==================================

result_counts = data["FTR"].value_counts()

result_labels = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

result_counts.index = [
    result_labels[x] for x in result_counts.index
]

result_percent = (
    result_counts / result_counts.sum() * 100
)

print("\nMATCH RESULT DISTRIBUTION")

for result, percent in result_percent.items():
    print(f"{result}: {percent:.2f}%")

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    x=result_percent.index,
    y=result_percent.values
)

plt.title("Premier League Match Result Distribution")
plt.xlabel("Match Result")
plt.ylabel("Percentage (%)")

plt.tight_layout()

plt.savefig(
    PROCESSED_DIR / "result_distribution.png",
    dpi=300
)

plt.show()

# ==================================
# 6. TEAM PERFORMANCE ANALYSIS
# ==================================

home = pd.DataFrame({
    "Team": data["HomeTeam"],
    "GF": data["FTHG"],
    "GA": data["FTAG"],
    "Result": data["FTR"].map({
        "H": "W",
        "D": "D",
        "A": "L"
    })
})

away = pd.DataFrame({
    "Team": data["AwayTeam"],
    "GF": data["FTAG"],
    "GA": data["FTHG"],
    "Result": data["FTR"].map({
        "H": "L",
        "D": "D",
        "A": "W"
    })
})

team_matches = pd.concat(
    [home, away],
    ignore_index=True
)

team_stats = team_matches.groupby("Team").agg(
    Played=("Result", "count"),
    Wins=("Result", lambda x: (x == "W").sum()),
    Draws=("Result", lambda x: (x == "D").sum()),
    Losses=("Result", lambda x: (x == "L").sum()),
    Goals_For=("GF", "sum"),
    Goals_Against=("GA", "sum")
)

team_stats["Points"] = (
    team_stats["Wins"] * 3
    + team_stats["Draws"]
)

team_stats["PPG"] = (
    team_stats["Points"] / team_stats["Played"]
)

team_stats["GF_per_Match"] = (
    team_stats["Goals_For"] / team_stats["Played"]
)

team_stats["GA_per_Match"] = (
    team_stats["Goals_Against"] / team_stats["Played"]
)

team_stats["Goal_Difference"] = (
    team_stats["Goals_For"]
    - team_stats["Goals_Against"]
)

# Sap xep theo tong diem
team_stats = team_stats.sort_values(
    "Points",
    ascending=False
)

print("\nTEAM PERFORMANCE")
print(team_stats.round(2))

team_stats.to_csv(
    PROCESSED_DIR / "team_statistics.csv"
)

# Top 10 teams by total goals
top_attack = team_stats.sort_values(
    "Goals_For",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_attack["Goals_For"],
    y=top_attack.index
)

plt.title("Top 10 Teams - Total Goals Scored")
plt.xlabel("Goals Scored")
plt.ylabel("Team")

plt.tight_layout()

plt.savefig(
    PROCESSED_DIR / "top_attack.png",
    dpi=300
)

plt.show()

# Top 10 teams by lowest goals conceded per match
top_defense = team_stats.sort_values(
    "GA_per_Match",
    ascending=True
).head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_defense["GA_per_Match"],
    y=top_defense.index
)

plt.title("Top 10 Teams - Lowest Goals Conceded Per Match")
plt.xlabel("Goals Conceded Per Match")
plt.ylabel("Team")

plt.tight_layout()

plt.savefig(
    PROCESSED_DIR / "top_defense.png",
    dpi=300
)

plt.show()

# ==================================
# 7. HOME ADVANTAGE ANALYSIS
# ==================================

home_win = (data["FTR"] == "H").sum()
draw = (data["FTR"] == "D").sum()
away_win = (data["FTR"] == "A").sum()

total = len(data)

home_win_pct = home_win / total * 100
draw_pct = draw / total * 100
away_win_pct = away_win / total * 100

print("\nHOME ADVANTAGE ANALYSIS")

print(f"Home Win: {home_win_pct:.2f}%")
print(f"Draw: {draw_pct:.2f}%")
print(f"Away Win: {away_win_pct:.2f}%")

print(
    f"Home teams earn a win in "
    f"{home_win_pct:.2f}% of matches."
)
