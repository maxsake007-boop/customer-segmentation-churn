# ============================================================
# app/main.py — Customer Segmentation & Churn Dashboard
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F1F5F9; font-size: 17px; }
    [data-testid="stSidebar"] { background-color: #1E293B; }
    [data-testid="stSidebar"] p { font-size: 16px !important; }

    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 28px;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 34px !important; font-weight: 700 !important; }

    h1 { color: #F1F5F9 !important; font-size: 38px !important; margin-bottom: 8px !important; }
    h2 { color: #F1F5F9 !important; font-size: 28px !important; }
    h3 { color: #F1F5F9 !important; font-size: 22px !important; }
    p  { font-size: 17px !important; color: #CBD5E1; line-height: 1.7; }

    .rec-card {
        background-color: #1E293B;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .rec-card.champions { border-left-color: #10B981; }
    .rec-card.loyal     { border-left-color: #2563EB; }
    .rec-card.atrisk    { border-left-color: #F59E0B; }
    .rec-card.lost      { border-left-color: #EF4444; }

    .rec-title { font-weight: 700; font-size: 24px; color: #F1F5F9; margin-bottom: 12px; }
    .rec-text  { font-size: 17px; color: #94A3B8; line-height: 1.9; }

    hr { border-color: #334155; margin: 28px 0; }

    .stDownloadButton button {
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/customers_segmented.csv")

df = load_data()

# ── GLOBAL STATS ──────────────────────────────────────────────
total_customers = len(df)
avg_monetary    = df["Monetary"].mean()
churn_rate      = df["Churned"].mean() * 100
high_risk_count = (df["ChurnRisk"] == "High").sum()

SEGMENT_COLORS = {
    "Champions":       "#10B981",
    "Loyal Customers": "#2563EB",
    "At Risk":         "#F59E0B",
    "Lost Customers":  "#EF4444"
}
RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 Customer Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Customer Segments", "Churn Risk"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Total Customers:** {total_customers:,}")
    st.markdown(f"**Churn Rate:** {churn_rate:.1f}%")
    st.markdown(f"**High Risk:** {high_risk_count:,}")
    st.markdown(f"**Avg Spend:** £{avg_monetary:,.0f}")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Customer Analytics Overview")
    st.markdown("Segmentation and churn analysis based on RFM model — Recency, Frequency, and Monetary value.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Avg Total Spend", f"£{avg_monetary:,.0f}")
    col3.metric("Churn Rate",      f"{churn_rate:.1f}%")
    col4.metric("High Churn Risk", f"{high_risk_count:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Customers by Segment")
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig1 = go.Figure(go.Bar(
            x=seg_counts["Segment"], y=seg_counts["Count"],
            marker_color=[SEGMENT_COLORS.get(s, "#2563EB") for s in seg_counts["Segment"]],
            text=seg_counts["Count"], textposition="outside",
            textfont=dict(size=16, color="#F1F5F9")
        ))
        fig1.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font=dict(color="#94A3B8", size=14),
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(size=14)),
            yaxis=dict(gridcolor="#1E293B"),
            height=420, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("Churn Risk Distribution")
        risk_counts = df["ChurnRisk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        risk_order = ["High", "Medium", "Low"]
        risk_counts["Risk"] = pd.Categorical(risk_counts["Risk"], categories=risk_order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk")

        fig2 = go.Figure(go.Bar(
            x=risk_counts["Risk"], y=risk_counts["Count"],
            marker_color=[RISK_COLORS.get(r, "#2563EB") for r in risk_counts["Risk"]],
            text=risk_counts["Count"], textposition="outside",
            textfont=dict(size=16, color="#F1F5F9")
        ))
        fig2.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font=dict(color="#94A3B8", size=14),
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(size=14)),
            yaxis=dict(gridcolor="#1E293B"),
            height=420, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Average Revenue per Customer by Segment")
    seg_monetary = df.groupby("Segment")["Monetary"].mean().sort_values(ascending=True).reset_index()

    fig3 = go.Figure(go.Bar(
        x=seg_monetary["Monetary"], y=seg_monetary["Segment"],
        orientation="h",
        marker_color=[SEGMENT_COLORS.get(s, "#2563EB") for s in seg_monetary["Segment"]],
        text=[f"£{v:,.0f}" for v in seg_monetary["Monetary"]],
        textposition="outside",
        textfont=dict(size=15, color="#F1F5F9")
    ))
    fig3.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font=dict(color="#94A3B8", size=14),
        xaxis=dict(gridcolor="#1E293B", tickprefix="£"),
        yaxis=dict(gridcolor="#1E293B", tickfont=dict(size=15)),
        height=380, margin=dict(l=0, r=120, t=20, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════════
elif page == "Customer Segments":
    st.title("Customer Segments")
    st.markdown("Each customer is assigned to a segment based on their purchasing behavior. Use the filter to explore each group.")
    st.markdown("---")

    selected_segment = st.selectbox(
        "Select Segment",
        ["All"] + sorted(df["Segment"].unique().tolist()),
        index=0
    )

    filtered = df if selected_segment == "All" else df[df["Segment"] == selected_segment]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers",     f"{len(filtered):,}")
    col2.metric("Avg Recency",   f"{filtered['Recency'].mean():.0f} days")
    col3.metric("Avg Frequency", f"{filtered['Frequency'].mean():.1f} orders")
    col4.metric("Avg Spend",     f"£{filtered['Monetary'].mean():,.0f}")

    st.markdown("---")

    st.subheader("Segment Recommendations")

    segment_recs = {
        "Champions": ("champions", "🏆 Champions — Your Most Valuable Customers",
            f"<strong>{len(df[df['Segment']=='Champions']):,} customers</strong> who buy frequently and recently with the highest spend "
            f"(avg £{df[df['Segment']=='Champions']['Monetary'].mean():,.0f}). "
            "These are the backbone of your business — treat them as VIPs. "
            "Reward them with early access to new products, exclusive loyalty discounts, and personal thank-you messages. "
            "Ask for reviews and referrals — Champions are your best brand ambassadors. "
            "Do NOT bombard them with generic promotions — they deserve personalized communication."),

        "Loyal Customers": ("loyal", "💙 Loyal Customers — Nurture and Grow",
            f"<strong>{len(df[df['Segment']=='Loyal Customers']):,} customers</strong> with recent purchases and consistent behavior "
            f"(avg £{df[df['Segment']=='Loyal Customers']['Monetary'].mean():,.0f}). "
            "They are reliable but haven't reached Champion level yet. "
            "Focus on increasing their order frequency and average basket size. "
            "Offer product recommendations based on past purchases, introduce a loyalty points program, "
            "and send personalized re-engagement emails after each purchase to encourage the next one."),

        "At Risk": ("atrisk", "⚠️ At Risk — Act Before It's Too Late",
            f"<strong>{len(df[df['Segment']=='At Risk']):,} customers</strong> who used to buy regularly but have gone quiet "
            f"(avg £{df[df['Segment']=='At Risk']['Monetary'].mean():,.0f}, last purchase ~{df[df['Segment']=='At Risk']['Recency'].mean():.0f} days ago). "
            "This is your highest priority retention group — they have real value but are drifting away. "
            "Send a 'We miss you' campaign with a time-limited discount (e.g. 15% off next order). "
            "Survey them to understand why they stopped buying. "
            "Every At Risk customer you save is worth far more than acquiring a new one."),

        "Lost Customers": ("lost", "❌ Lost Customers — Win-Back or Let Go",
            f"<strong>{len(df[df['Segment']=='Lost Customers']):,} customers</strong> who haven't purchased in a very long time "
            f"(avg {df[df['Segment']=='Lost Customers']['Recency'].mean():.0f} days ago, avg £{df[df['Segment']=='Lost Customers']['Monetary'].mean():,.0f}). "
            "Run a final win-back campaign with a strong offer (e.g. 20-25% off). "
            "If they don't respond, remove them from regular marketing lists to reduce costs and improve email deliverability. "
            "Focus your budget on segments with higher potential ROI.")
    }

    if selected_segment == "All":
        for seg, (css_class, title, text) in segment_recs.items():
            st.markdown(f"""
            <div class="rec-card {css_class}">
                <div class="rec-title">{title}</div>
                <div class="rec-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)
    elif selected_segment in segment_recs:
        css_class, title, text = segment_recs[selected_segment]
        st.markdown(f"""
        <div class="rec-card {css_class}">
            <div class="rec-title">{title}</div>
            <div class="rec-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader(f"Customer List — {selected_segment}")

    display_df = filtered[[
        "Customer ID", "Segment", "Recency", "Frequency",
        "Monetary", "RFM_Score", "ChurnRisk", "ChurnProbability"
    ]].copy()
    display_df.columns = [
        "Customer ID", "Segment", "Recency (days)", "Orders",
        "Total Spend (£)", "RFM Score", "Churn Risk", "Churn Probability"
    ]
    display_df["Total Spend (£)"] = display_df["Total Spend (£)"].round(2)
    display_df = display_df.sort_values("Total Spend (£)", ascending=False).reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Segment Data as CSV",
        data=csv,
        file_name=f"segment_{selected_segment.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )


# ════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN RISK
# ════════════════════════════════════════════════════════════════
elif page == "Churn Risk":
    st.title("Churn Risk Analysis")
    st.markdown("XGBoost model predicts the probability of each customer churning. Accuracy: **91.7%** · ROC AUC: **98.5%**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk",   f"{(df['ChurnRisk']=='High').sum():,}",   f"{(df['ChurnRisk']=='High').mean()*100:.1f}%")
    col2.metric("Medium Risk", f"{(df['ChurnRisk']=='Medium').sum():,}", f"{(df['ChurnRisk']=='Medium').mean()*100:.1f}%")
    col3.metric("Low Risk",    f"{(df['ChurnRisk']=='Low').sum():,}",    f"{(df['ChurnRisk']=='Low').mean()*100:.1f}%")
    col4.metric("Avg Churn Probability", f"{df['ChurnProbability'].mean():.1%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Probability Distribution")
        fig4 = go.Figure(go.Histogram(
            x=df["ChurnProbability"], nbinsx=40,
            marker_color="#2563EB",
            marker_line_color="#0F172A", marker_line_width=1
        ))
        fig4.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font=dict(color="#94A3B8", size=14),
            xaxis=dict(gridcolor="#1E293B", title="Churn Probability", tickformat=".0%"),
            yaxis=dict(gridcolor="#1E293B", title="Number of Customers"),
            height=400, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_right:
        st.subheader("Churn Risk by Segment")
        risk_seg = df.groupby(["Segment", "ChurnRisk"]).size().reset_index(name="Count")

        fig5 = px.bar(
            risk_seg, x="Segment", y="Count", color="ChurnRisk",
            color_discrete_map=RISK_COLORS, barmode="stack"
        )
        fig5.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font=dict(color="#94A3B8", size=14),
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(size=13)),
            yaxis=dict(gridcolor="#1E293B", title="Number of Customers"),
            legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1, font=dict(size=14)),
            height=400, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    st.subheader("Customer Churn Risk List")
    st.markdown("Filter by risk level. Export the list to use in your CRM or email marketing tool.")

    risk_filter = st.selectbox(
        "Filter by Churn Risk",
        ["High", "Medium", "Low", "All"],
        index=0
    )

    risk_df = df.copy() if risk_filter == "All" else df[df["ChurnRisk"] == risk_filter].copy()

    display_risk = risk_df[[
        "Customer ID", "Segment", "Recency", "Frequency",
        "Monetary", "ChurnProbability", "ChurnRisk"
    ]].copy()
    display_risk.columns = [
        "Customer ID", "Segment", "Recency (days)", "Orders",
        "Total Spend (£)", "Churn Probability", "Churn Risk"
    ]
    display_risk["Total Spend (£)"]   = display_risk["Total Spend (£)"].round(2)
    display_risk["Churn Probability"] = display_risk["Churn Probability"].apply(lambda x: f"{x:.1%}")
    display_risk = display_risk.sort_values("Churn Probability", ascending=False).reset_index(drop=True)

    st.dataframe(display_risk, use_container_width=True, hide_index=True)

    csv_risk = risk_df[[
        "Customer ID", "Segment", "Recency", "Frequency",
        "Monetary", "ChurnProbability", "ChurnRisk"
    ]].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Churn Risk List as CSV",
        data=csv_risk,
        file_name="churn_risk_customers.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("What Drives Churn — SHAP Feature Importance")
    st.markdown("The chart below shows which factors have the biggest impact on the churn prediction model.")
    st.image("outputs/shap_bar.png", use_container_width=True)