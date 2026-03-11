# ============================================================
# scripts/run_features.py — RFM Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────
df = pd.read_csv("data/processed/online_retail_clean.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("=" * 55)
print("RFM FEATURE ENGINEERING")
print("=" * 55)
print(f"Loaded: {df.shape[0]:,} rows")
print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")

# ── 2. REFERENCE DATE ─────────────────────────────────────────
# One day after last transaction — standard RFM practice
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"\n[INFO] Reference date: {reference_date.date()}")

# ── 3. BUILD RFM TABLE ────────────────────────────────────────
rfm = df.groupby("Customer ID").agg(
    Recency   = ("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency = ("Invoice",     "nunique"),
    Monetary  = ("TotalPrice",  "sum")
).reset_index()

rfm["Monetary"] = rfm["Monetary"].round(2)

print(f"\n[INFO] RFM table built: {len(rfm):,} customers")
print(f"\nRFM Sample:")
print(rfm.head(10).to_string())

# ── 4. RFM STATISTICS ─────────────────────────────────────────
print(f"\n" + "=" * 55)
print("RFM STATISTICS")
print("=" * 55)
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())

# ── 5. RFM SCORES (1-5) ───────────────────────────────────────
# Recency: lower is better → higher score for lower recency
rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1, 2, 3, 4, 5]).astype(int)

rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

print(f"\n[INFO] RFM scores calculated (1-5 scale).")
print(f"RFM Score range: {rfm['RFM_Score'].min()} — {rfm['RFM_Score'].max()}")

# ── 6. CHURN LABEL ────────────────────────────────────────────
# Customers who haven't purchased in last 90 days = churned
CHURN_THRESHOLD = 90
rfm["Churned"] = (rfm["Recency"] > CHURN_THRESHOLD).astype(int)

churned_count = rfm["Churned"].sum()
print(f"\n[INFO] Churn threshold: {CHURN_THRESHOLD} days")
print(f"[INFO] Churned customers: {churned_count:,} ({churned_count/len(rfm)*100:.1f}%)")
print(f"[INFO] Active customers:  {len(rfm) - churned_count:,} ({(len(rfm)-churned_count)/len(rfm)*100:.1f}%)")

# ── 7. ADDITIONAL FEATURES ────────────────────────────────────
# Avg order value per customer
customer_aov = df.groupby("Customer ID")["TotalPrice"].mean().round(2)
rfm["AvgOrderValue"] = rfm["Customer ID"].map(customer_aov)

# Number of unique products purchased
customer_products = df.groupby("Customer ID")["StockCode"].nunique()
rfm["UniqueProducts"] = rfm["Customer ID"].map(customer_products)

# Number of countries (should be 1 for most)
customer_countries = df.groupby("Customer ID")["Country"].nunique()
rfm["UniqueCountries"] = rfm["Customer ID"].map(customer_countries)

print(f"\n[INFO] Additional features added: AvgOrderValue, UniqueProducts, UniqueCountries")

# ── 8. FINAL OVERVIEW ─────────────────────────────────────────
print(f"\n" + "=" * 55)
print("FINAL FEATURE TABLE")
print("=" * 55)
print(f"Shape: {rfm.shape[0]:,} rows × {rfm.shape[1]} columns")
print(f"Columns: {rfm.columns.tolist()}")
print(f"\nMissing values: {rfm.isnull().sum().sum()}")

# ── 9. VISUALIZATIONS ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(rfm["Recency"],   bins=50, color="#2563EB", edgecolor="white")
axes[0].set_title("Recency Distribution")
axes[0].set_xlabel("Days since last purchase")

axes[1].hist(rfm["Frequency"], bins=50, color="#2563EB", edgecolor="white")
axes[1].set_title("Frequency Distribution")
axes[1].set_xlabel("Number of orders")

axes[2].hist(rfm["Monetary"],  bins=50, color="#2563EB", edgecolor="white")
axes[2].set_title("Monetary Distribution")
axes[2].set_xlabel("Total spend (£)")

plt.tight_layout()
plt.savefig("outputs/rfm_distributions.png", dpi=150)
plt.show()
print("\n[INFO] Chart saved → outputs/rfm_distributions.png")

# ── 10. SAVE ──────────────────────────────────────────────────
rfm.to_csv("data/processed/rfm_features.csv", index=False)
print(f"[SUCCESS] RFM features saved → data/processed/rfm_features.csv")