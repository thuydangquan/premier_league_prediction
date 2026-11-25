
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

files = list(RAW_DIR.glob("EPL_*.csv"))

for file in files:

    print("\n" + "=" * 50)
    print("SEASON:", file.stem)
    print("=" * 50)

    df = pd.read_csv(file)

    # 1. So luong tran dau
    print("\nNumber of matches:", len(df))

    # 2. Xem 5 dong dau tien
    print("\nFirst 5 matches:")
    print(df.head())

    # 3. Kieu du lieu
    print("\nData information:")
    df.info()

    # 4. Kiem tra gia tri thieu
    print("\nMissing values:")
    print(df.isnull().sum())

    # 5. Thong ke ket qua
    if "FTR" in df.columns:
        print("\nMatch results:")
        print(df["FTR"].value_counts())

print("\nInspection completed!")
