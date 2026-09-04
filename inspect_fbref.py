"""
Quick inspection - check column names and a sample row from the two
FBref dataset sources before writing the merge script.

Run from the project root:
    python3 inspect_fbref.py
"""
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 70)
print("cleaned_2023-24.csv (older multi-season dataset)")
print("=" * 70)
df1 = pd.read_csv("data/raw/fbref/cleaned_2023-24.csv")
print(f"Shape: {df1.shape}")
print("Columns:", list(df1.columns))
print("\nSample row:")
print(df1.head(2).to_string())

print("\n\n" + "=" * 70)
print("players_data-2025_2026.csv (newer single-season dataset)")
print("=" * 70)
df2 = pd.read_csv("data/raw/fbref/players_data-2025_2026.csv")
print(f"Shape: {df2.shape}")
print("Columns:", list(df2.columns))
print("\nSample row:")
print(df2.head(2).to_string())