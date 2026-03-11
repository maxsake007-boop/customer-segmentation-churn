# ============================================================
# scripts/run_model.py — KMeans Clustering + Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

# ── 1. LOAD DATA ──────────────────────────────────────────────
rfm = pd.read_csv("data/processed/rfm_features.csv")

print("=" * 55)
print("MODEL TRAINING — SEGMENTATION + CHURN")
print("=" * 55)
print(f"Loaded: {len(rfm):,} customers")

# ── 2. LOG TRANSFORMATION ─────────────────────────────────────
# Reduce skewness for better clustering
rfm["Log_Recency"]   = np.log1p(rfm["Recency"])
rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])
rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"])

print("[INFO] Log transformation applied to RFM columns.")

# ── 3. SCALE FEATURES ─────────────────────────────────────────
CLUSTER_FEATURES = ["Log_Recency", "Log_Frequency", "Log_Monetary"]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(rfm[CLUSTER_FEATURES])

print("[INFO] Features scaled.")

# ── 4. ELBOW METHOD — find optimal K ─────────────────────────
print("\n[INFO] Running Elbow Method...")
inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker="o", color="#2563EB", linewidth=2)
plt.title("Elbow Method — Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(K_range)
plt.tight_layout()
plt.savefig("outputs/elbow_method.png", dpi=150)
plt.show()
print("[INFO] Elbow chart saved → outputs/elbow_method.png")

# ── 5. TRAIN KMEANS ───────────────────────────────────────────
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)
print(f"\n[INFO] KMeans trained with {N_CLUSTERS} clusters.")

# ── 6. CLUSTER ANALYSIS ───────────────────────────────────────
print("\n" + "=" * 55)
print("CLUSTER ANALYSIS")
print("=" * 55)

cluster_summary = rfm.groupby("Cluster").agg(
    Customers   = ("Customer ID", "count"),
    Avg_Recency = ("Recency",     "mean"),
    Avg_Frequency=("Frequency",  "mean"),
    Avg_Monetary = ("Monetary",  "mean"),
    Churn_Rate  = ("Churned",    "mean")
).round(2)

cluster_summary["Churn_Rate"] = (cluster_summary["Churn_Rate"] * 100).round(1)
cluster_summary["Avg_Monetary"] = cluster_summary["Avg_Monetary"].apply(lambda x: f"£{x:,.0f}")
print(cluster_summary.to_string())

# ── 7. NAME CLUSTERS ─────────────────────────────────────────
# Based on RFM values — assign business labels
rfm_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

labels = {}
for cluster in rfm_means.index:
    r = rfm_means.loc[cluster, "Recency"]
    f = rfm_means.loc[cluster, "Frequency"]
    m = rfm_means.loc[cluster, "Monetary"]

    if r < 60 and f > 5:
        labels[cluster] = "Champions"
    elif r < 90 and f >= 3:
        labels[cluster] = "Loyal Customers"
    elif r > 300:
        labels[cluster] = "Lost Customers"
    else:
        labels[cluster] = "At Risk"

rfm["Segment"] = rfm["Cluster"].map(labels)
print(f"\n[INFO] Cluster labels assigned:")
print(rfm.groupby(["Cluster", "Segment"]).size().reset_index(name="Count").to_string())

# ── 8. SAVE SCALER + KMEANS ───────────────────────────────────
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
print("\n[INFO] Scaler and KMeans saved.")

# ── 9. CHURN PREDICTION — XGBoost ────────────────────────────
print("\n" + "=" * 55)
print("CHURN PREDICTION — XGBoost")
print("=" * 55)

CHURN_FEATURES = [
    "Frequency", "Monetary",
    "F_Score", "M_Score", "RFM_Score",
    "AvgOrderValue", "UniqueProducts"
]
TARGET = "Churned"

X = rfm[CHURN_FEATURES]
y = rfm[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    verbosity=0,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
pred  = xgb.predict(X_test)
proba = xgb.predict_proba(X_test)[:, 1]

# ── 10. CHURN METRICS ─────────────────────────────────────────
print(f"\nAccuracy:  {accuracy_score(y_test, pred):.2%}")
print(f"F1 Score:  {f1_score(y_test, pred):.2%}")
print(f"ROC AUC:   {roc_auc_score(y_test, proba):.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Active", "Churned"]))

# ── 11. CHURN PROBABILITY FOR ALL CUSTOMERS ──────────────────
rfm["ChurnProbability"] = xgb.predict_proba(rfm[CHURN_FEATURES])[:, 1].round(3)
rfm["ChurnRisk"] = pd.cut(
    rfm["ChurnProbability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
)

print(f"\n[INFO] Churn risk distribution:")
print(rfm["ChurnRisk"].value_counts().to_string())

# ── 12. SAVE CHURN MODEL ──────────────────────────────────────
joblib.dump(xgb, "models/churn_model.pkl")
print("\n[INFO] Churn model saved → models/churn_model.pkl")

# ── 13. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Cluster sizes
seg_counts = rfm["Segment"].value_counts()
axes[0].bar(seg_counts.index, seg_counts.values, color="#2563EB")
axes[0].set_title("Customers by Segment")
axes[0].set_ylabel("Number of Customers")

# Avg Monetary by Segment
seg_monetary = rfm.groupby("Segment")["Monetary"].mean().sort_values()
axes[1].barh(seg_monetary.index, seg_monetary.values, color="#2563EB")
axes[1].set_title("Avg Revenue by Segment")
axes[1].set_xlabel("Avg Total Spend (£)")

# Churn Risk distribution
risk_counts = rfm["ChurnRisk"].value_counts()
colors = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
axes[2].bar(
    risk_counts.index,
    risk_counts.values,
    color=[colors[r] for r in risk_counts.index]
)
axes[2].set_title("Churn Risk Distribution")
axes[2].set_ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("outputs/model_results.png", dpi=150)
plt.show()
print("[INFO] Chart saved → outputs/model_results.png")

# ── 14. SAVE FINAL DATASET ────────────────────────────────────
rfm.to_csv("data/processed/customers_segmented.csv", index=False)
print(f"\n[SUCCESS] Segmented customers saved → data/processed/customers_segmented.csv")
print(f"Final columns: {rfm.columns.tolist()}")


# ── 15. SHAP — Feature Importance ────────────────────────────
import shap

print("\n" + "=" * 55)
print("SHAP — FEATURE IMPORTANCE")
print("=" * 55)

explainer   = shap.Explainer(xgb)
shap_values = explainer(X_train)

# Summary plot — overall feature importance
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP — Feature Impact on Churn Prediction", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] SHAP summary saved → outputs/shap_summary.png")

# Bar plot — mean absolute SHAP values
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP — Mean Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] SHAP bar chart saved → outputs/shap_bar.png")