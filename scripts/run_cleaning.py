# ============================================================
# scripts/run_cleaning.py — Data Cleaning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("data/raw/online_retail_II.csv", encoding="utf-8")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# ── 2. BASIC INFO ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("DATA TYPES & MISSING VALUES")
print("=" * 55)
df.info()

missing = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing %": (df.isna().sum() / len(df) * 100).round(2)
})
print(missing[missing["Missing Count"] > 0].sort_values("Missing %", ascending=False))

# ── 3. DUPLICATES ─────────────────────────────────────────────
print(f"\nDuplicate rows: {df.duplicated().sum():,}")

# ── 4. FIX DATA TYPES ─────────────────────────────────────────
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
print("\n[INFO] Data types fixed.")

# ── 5. REMOVE CANCELLATIONS ───────────────────────────────────
before = len(df)
df = df[~df["Invoice"].astype(str).str.startswith("C")]
print(f"[INFO] Removed {before - len(df):,} cancellation rows.")

# ── 6. REMOVE MISSING Customer ID ────────────────────────────
before = len(df)
df = df.dropna(subset=["Customer ID"])
print(f"[INFO] Removed {before - len(df):,} rows with no Customer ID.")

# ── 7. REMOVE NEGATIVE / ZERO VALUES ─────────────────────────
before = len(df)
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
print(f"[INFO] Removed {before - len(df):,} rows with negative/zero Quantity or Price.")

# ── 8. REMOVE DUPLICATES ──────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"[INFO] Removed {before - len(df):,} duplicate rows.")

# ── 9. FIX Customer ID TYPE ───────────────────────────────────
df["Customer ID"] = df["Customer ID"].astype(int)

# ── 10. ADD TOTAL PRICE COLUMN ────────────────────────────────
df["TotalPrice"] = df["Quantity"] * df["Price"]
print("[INFO] TotalPrice column added.")

# ── 11. SORT BY DATE ──────────────────────────────────────────
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# ── 12. FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY AFTER CLEANING")
print("=" * 55)
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")
print(f"Unique products: {df['StockCode'].nunique():,}")
print(f"Countries: {df['Country'].nunique()}")
print(f"\nMissing values:")
final_missing = df.isnull().sum()
print(final_missing[final_missing > 0] if final_missing.sum() > 0 else "  None")

# ── 13. SAVE ──────────────────────────────────────────────────
df.to_csv("data/processed/online_retail_clean.csv", index=False)
print(f"\n[SUCCESS] Clean dataset saved → data/processed/online_retail_clean.csv")

# ── 14. QUICK VISUAL CHECK ────────────────────────────────────
daily = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()

plt.figure(figsize=(14, 4))
plt.plot(daily.index, daily.values, color="#2563EB", linewidth=1)
plt.title("Daily Revenue — After Cleaning")
plt.xlabel("Date")
plt.ylabel("Revenue (£)")
plt.tight_layout()
plt.savefig("outputs/cleaning_summary.png", dpi=150)
plt.show()
print("[INFO] Chart saved → outputs/cleaning_summary.png")