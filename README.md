# Customer Segmentation & Churn Prediction Dashboard

An interactive Streamlit application that segments customers into 
behavior-based groups and predicts churn risk — helping businesses 
retain valuable customers and prioritize marketing efforts.

## Demo
> Add screenshots here

## Features
- RFM analysis — Recency, Frequency, Monetary segmentation
- 4 customer segments: Champions, Loyal, At Risk, Lost
- Churn prediction with probability score for every customer
- SHAP analysis — understand what drives churn
- Export high-risk customers to CSV for CRM
- Business recommendations for each segment

## Tech Stack
- **ML Models:** KMeans, XGBoost
- **Dashboard:** Streamlit, Plotly
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, SHAP

## Model Performance
- **Accuracy:** 91.7%
- **ROC AUC:** 98.5%
- **Dataset:** UK Online Retail II — 5,878 customers

## Project Structure
```
ПРОЕКТ 2/
├── app/
│   └── main.py                  # Streamlit dashboard
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Cleaned and featured data
├── models/
│   ├── churn_model.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/                     # Visualizations and plots
├── scripts/
│   ├── run_cleaning.py
│   ├── run_features.py
│   └── run_model.py
├── src/                         # Helper functions
├── requirements.txt
└── .gitignore
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-segmentation-churn.git
cd customer-segmentation-churn
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the dashboard
```bash
streamlit run app/main.py
```

## Author
Max Narbekov — Data Analyst / Data Scientist