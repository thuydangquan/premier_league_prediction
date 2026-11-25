
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# 1. SETUP
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "EPL_all_seasons.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

# =====================================
# 2. LOAD DATA
# =====================================

data = pd.read_csv(DATA_PATH)

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").copy()

print("Historical data loaded:", len(data))

# =====================================
# 3. CREATE HOME TEAM HISTORY
# =====================================

home = pd.DataFrame({
    "Date": data["Date"],
    "Season": data["Season"],
    "Team": data["HomeTeam"],
    "Opponent": data["AwayTeam"],
    "Goals_For": data["FTHG"],
    "Goals_Against": data["FTAG"],
    "Result": data["FTR"].map({
        "H": "W",
        "D": "D",
        "A": "L"
    })
})

# =====================================
# 4. CREATE AWAY TEAM HISTORY
# =====================================

away = pd.DataFrame({
    "Date": data["Date"],
    "Season": data["Season"],
    "Team": data["AwayTeam"],
    "Opponent": data["HomeTeam"],
    "Goals_For": data["FTAG"],
    "Goals_Against": data["FTHG"],
    "Result": data["FTR"].map({
        "H": "L",
        "D": "D",
        "A": "W"
    })
})

# =====================================
# 5. COMBINE TEAM HISTORY
# =====================================

team_history = pd.concat(
    [home, away],
    ignore_index=True
)

team_history["Points"] = team_history["Result"].map({
    "W": 3,
    "D": 1,
    "L": 0
})

team_history = team_history.sort_values(
    ["Team", "Date"]
).reset_index(drop=True)

# =====================================
# 6. SAVE TEAM HISTORY
# =====================================

history_path = OUTPUT_DIR / "team_history.csv"

team_history.to_csv(
    history_path,
    index=False
)

print("\nTeam history created!")
print("Total team-match records:", len(team_history))

print("\nSample data:")
print(team_history.head(10))

print("\nSaved to:", history_path)

# =====================================
# 7. CALCULATE RECENT FORM
# =====================================

def weighted_average(values):
    n = len(values)

    weights = range(1, n + 1)

    weighted_sum = sum(
        value * weight
        for value, weight in zip(values, weights)
    )

    return weighted_sum / sum(weights)


# =====================================
# 8. CALCULATE FORM 5 AND FORM 10
# =====================================

team_history = team_history.sort_values(
    ["Team", "Date"]
).copy()

# Shift(1): chi su dung cac tran truoc tran hien tai
for column in ["Points", "Goals_For", "Goals_Against"]:

    previous = team_history.groupby("Team")[column].shift(1)

    for window in [5, 10]:

        feature_name = f"{column}_{window}"

        team_history[feature_name] = (
            previous.groupby(team_history["Team"])
            .rolling(window=window, min_periods=1)
            .apply(weighted_average, raw=True)
            .reset_index(level=0, drop=True)
        )


# =====================================
# 9. COMBINED FORM
# =====================================

team_history["Form_5"] = team_history["Points_5"]
team_history["Form_10"] = team_history["Points_10"]

team_history["Combined_Form"] = (
    0.65 * team_history["Form_5"]
    + 0.35 * team_history["Form_10"]
)

# =====================================
# 10. SAVE RESULTS
# =====================================

team_history.to_csv(
    history_path,
    index=False
)

print("\nRECENT FORM ANALYSIS")

print(
    team_history[
        [
            "Team",
            "Date",
            "Points",
            "Form_5",
            "Form_10",
            "Combined_Form"
        ]
    ].tail(15).round(2)
)

print("\nRecent form calculated successfully!")

# =====================================
# 11. ATTACK AND DEFENSE ANALYSIS
# =====================================

team_history["Attack_Form"] = (
    0.65 * team_history["Goals_For_5"]
    + 0.35 * team_history["Goals_For_10"]
)

team_history["Defense_Form"] = (
    0.65 * team_history["Goals_Against_5"]
    + 0.35 * team_history["Goals_Against_10"]
)

team_history["Goal_Difference_Form"] = (
    team_history["Attack_Form"]
    - team_history["Defense_Form"]
)

# =====================================
# 12. LATEST TEAM PERFORMANCE
# =====================================

latest_form = (
    team_history[
        team_history["Season"] == "2026/27"
    ]
    .sort_values("Date")
    .groupby("Team")
    .tail(1)
    .copy()
)

latest_form = latest_form.sort_values(
    "Combined_Form",
    ascending=False
)

columns_to_show = [
    "Team",
    "Date",
    "Form_5",
    "Form_10",
    "Combined_Form",
    "Attack_Form",
    "Defense_Form",
    "Goal_Difference_Form"
]

print("\nLATEST TEAM PERFORMANCE")

print(
    latest_form[columns_to_show]
    .round(2)
    .to_string(index=False)
)

# Save latest performance
latest_form[columns_to_show].to_csv(
    OUTPUT_DIR / "latest_team_performance.csv",
    index=False
)

print("\nLatest performance saved successfully!")

# =====================================
# 13. VISUALIZE TEAM FORM OVER TIME
# =====================================

season_data = team_history[
    team_history["Season"] == "2026/27"
].copy()

teams = ["Arsenal", "Man City", "Liverpool"]

form_features = {
    "Form_5": "Recent Form - Last 5 Matches",
    "Combined_Form": "Combined Form - 65% Form 5 + 35% Form 10"
}

for feature, title in form_features.items():

    plt.figure(figsize=(12, 6))

    for team in teams:

        team_data = season_data[
            season_data["Team"] == team
        ].sort_values("Date")

        if team_data.empty:
            print(f"Warning: {team} not found!")
            continue

        plt.plot(
            team_data["Date"],
            team_data[feature],
            marker="o",
            markersize=4,
            label=team
        )

    plt.title(title)
    plt.xlabel("Match Date")
    plt.ylabel("Form Score (0-3)")

    plt.ylim(0, 3)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = feature.lower() + "_comparison.png"

    plt.savefig(
    OUTPUT_DIR / filename,
    dpi=300
)
    

    plt.show()

print("\nForm visualization completed!")