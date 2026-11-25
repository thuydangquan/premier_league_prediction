
import pandas as pd
import requests
from pathlib import Path

# 1. Xac dinh thu muc project
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# 2. Danh sach mua giai
seasons = {
    "2024_2025": "2425",
    "2025_2026": "2526",
    "2026_2027": "2627"
}

# 3. Dia chi nguon du lieu
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# 4. Tai tung mua giai
for season_name, season_code in seasons.items():

    url = f"{BASE_URL}/{season_code}/E0.csv"

    print(f"\nDownloading {season_name}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Doc du lieu CSV
        from io import BytesIO

        df = pd.read_csv(
            BytesIO(response.content),
            encoding="utf-8-sig"
        )

        # Kiem tra du lieu
        required_columns = [
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            print("Missing columns:", missing)
            continue

        # Luu file
        file_path = RAW_DIR / f"EPL_{season_name}.csv"

        df.to_csv(file_path, index=False)

        print("Downloaded successfully!")
        print("Number of matches:", len(df))
        print("Saved at:", file_path)

    except Exception as e:
        print("Error:", e)

print("\nData collection completed!")
