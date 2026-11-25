
import pandas as pd
import numpy as np

from pathlib import Path

# ==========================================
# 1. SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "EPL_all_seasons.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

data = pd.read_csv(DATA_PATH)

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)

print("Data loaded:", len(data), "matches")


# ==========================================
# 3. WEIGHTED FORM FUNCTION
# ==========================================

def weighted_average(values):

    if len(values) == 0:
        return np.nan

    weights = np.arange(1, len(values) + 1)

    return np.average(values, weights=weights)


def calculate_form(history, column, window):

    recent = history[-window:]

    values = [
        match[column]
        for match in recent
    ]

    return weighted_average(values)


# ==========================================
# 4. TEAM HISTORICAL DATABASE
# ==========================================

team_history = {}

feature_rows = []


# ==========================================
# 5. PROCESS MATCHES CHRONOLOGICALLY
# ==========================================

for date, daily_matches in data.groupby("Date", sort=True):

    daily_updates = []

    for _, match in daily_matches.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_history = team_history.get(home, [])
        away_history = team_history.get(away, [])

        # ----------------------------------
        # HOME TEAM FEATURES
        # ----------------------------------

        home_f5 = calculate_form(
            home_history, "Points", 5
        )

        home_f10 = calculate_form(
            home_history, "Points", 10
        )

        home_attack5 = calculate_form(
            home_history, "Goals_For", 5
        )

        home_attack10 = calculate_form(
            home_history, "Goals_For", 10
        )

        home_defense5 = calculate_form(
            home_history, "Goals_Against", 5
        )

        home_defense10 = calculate_form(
            home_history, "Goals_Against", 10
        )

        # ----------------------------------
        # AWAY TEAM FEATURES
        # ----------------------------------

        away_f5 = calculate_form(
            away_history, "Points", 5
        )

        away_f10 = calculate_form(
            away_history, "Points", 10
        )

        away_attack5 = calculate_form(
            away_history, "Goals_For", 5
        )

        away_attack10 = calculate_form(
            away_history, "Goals_For", 10
        )

        away_defense5 = calculate_form(
            away_history, "Goals_Against", 5
        )

        away_defense10 = calculate_form(
            away_history, "Goals_Against", 10
        )

        # ----------------------------------
        # COMBINED FORM
        # ----------------------------------

        home_combined = (
            0.65 * home_f5 + 0.35 * home_f10
            if pd.notna(home_f5) and pd.notna(home_f10)
            else np.nan
        )

        away_combined = (
            0.65 * away_f5 + 0.35 * away_f10
            if pd.notna(away_f5) and pd.notna(away_f10)
            else np.nan
        )

        # ----------------------------------
        # ATTACK AND DEFENSE
        # ----------------------------------

        home_attack = (
            0.65 * home_attack5 + 0.35 * home_attack10
            if pd.notna(home_attack5) and pd.notna(home_attack10)
            else np.nan
        )

        away_attack = (
            0.65 * away_attack5 + 0.35 * away_attack10
            if pd.notna(away_attack5) and pd.notna(away_attack10)
            else np.nan
        )

        home_defense = (
            0.65 * home_defense5 + 0.35 * home_defense10
            if pd.notna(home_defense5) and pd.notna(home_defense10)
            else np.nan
        )

        away_defense = (
            0.65 * away_defense5 + 0.35 * away_defense10
            if pd.notna(home_defense5) and pd.notna(home_defense10)
            else np.nan
        )

        # ----------------------------------
        # BUILD FEATURE ROW
        # ----------------------------------

        row = {
            "Date": date,
            "Season": match["Season"],

            "HomeTeam": home,
            "AwayTeam": away,

            "Home_Prior_Matches": len(home_history),
            "Away_Prior_Matches": len(away_history),

            "Home_Form_5": home_f5,
            "Home_Form_10": home_f10,
            "Home_Combined_Form": home_combined,

            "Away_Form_5": away_f5,
            "Away_Form_10": away_f10,
            "Away_Combined_Form": away_combined,

            "Home_Attack_Form": home_attack,
            "Home_Defense_Form": home_defense,

            "Away_Attack_Form": away_attack,
            "Away_Defense_Form": away_defense,

            "Form_Difference": (
                home_combined - away_combined
                if pd.notna(home_combined)
                and pd.notna(away_combined)
                else np.nan
            ),

            "Attack_Difference": (
                home_attack - away_attack
                if pd.notna(home_attack)
                and pd.notna(away_attack)
                else np.nan
            ),

            "Defense_Difference": (
                home_defense - away_defense
                if pd.notna(home_defense)
                and pd.notna(away_defense)
                else np.nan
            ),

            # Target
            "FTR": match["FTR"],

            "FTHG": match["FTHG"],
            "FTAG": match["FTAG"]
        }

        feature_rows.append(row)

        # Save results for updating history later
        daily_updates.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": match["FTHG"],
            "FTAG": match["FTAG"],
            "FTR": match["FTR"]
        })

    # ----------------------------------
    # UPDATE HISTORY AFTER ALL MATCHES
    # ON THE SAME DATE ARE PROCESSED
    # ----------------------------------

    for match in daily_updates:

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        if home not in team_history:
            team_history[home] = []

        if away not in team_history:
            team_history[away] = []

        if match["FTR"] == "H":
            home_points, away_points = 3, 0

        elif match["FTR"] == "A":
            home_points, away_points = 0, 3

        else:
            home_points, away_points = 1, 1

        team_history[home].append({
            "Points": home_points,
            "Goals_For": match["FTHG"],
            "Goals_Against": match["FTAG"]
        })

        team_history[away].append({
            "Points": away_points,
            "Goals_For": match["FTAG"],
            "Goals_Against": match["FTHG"]
        })


# ==========================================
# 6. CREATE FINAL DATASET
# ==========================================

features = pd.DataFrame(feature_rows)

features = features.sort_values(
    ["Date", "HomeTeam"]
).reset_index(drop=True)


# ==========================================
# 7. SAVE DATASET
# ==========================================

output_path = OUTPUT_DIR / "EPL_features.csv"

features.to_csv(output_path, index=False)


# ==========================================
# 8. INSPECT RESULTS
# ==========================================

print("\nFEATURE ENGINEERING COMPLETED!")

print("Total matches:", len(features))

print("\nDataset columns:")
print(features.columns.tolist())

print("\nFirst 5 rows:")

print(
    features.head().to_string(index=False)
)

print("\nMissing values:")

print(
    features.isnull().sum()
)

print("\nSaved to:", output_path)
