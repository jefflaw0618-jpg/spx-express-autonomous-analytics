# 📦 SPX Express - Autonomous Analytics & Prediction Project

This project was built as part of the portfolio development for the **Data Analytics Intern - SPX Express** role. It demonstrates end-to-end data engineering, statistical hypothesis testing, predictive machine learning, and the deployment of a highly stylized autonomous dashboard.

## 🚀 Live Demo
> **[View Autonomous Dashboard](https://spx-express-autonomous-analytics.onrender.com)**

## 🎯 Project Overview

The core objective of this project is to identify operational gaps in last-mile delivery operations and provide data-driven AI tools to mitigate delays.

1. **Data Engineering & Regex Parsing:**
   Processed a real-world dataset of 45,000+ delivery records. Engineered geographic distances using the **Haversine Formula** (calculating exact kilometers between restaurant and delivery GPS coordinates). Used **Regex** to parse complex strings like `PUNERES20DEL01` into discrete components (`City_Code`, `Restaurant_No`, `Delivery_Sequence`).

2. **Statistical Rigor & Hypothesis Testing:**
   Conducted rigorous A/B and hypothesis testing to validate operational assumptions. Specifically utilized **Spearman Rank Correlation** for monotonic relationships involving ordinal data (e.g., Driver Rating vs Delivery Time), demonstrating advanced understanding of non-parametric statistics.

3. **Predictive Modeling & Principle of Parsimony:**
   Trained 6 separate machine learning models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost). 
   - Implemented an automated **Forward Selection algorithm** based on the **Principle of Parsimony (Occam's Razor)**.
   - Mathematically proved that 3 variables (`City_Code`, `Type_of_order`, `Type_of_vehicle`) contributed `< 0.01` to the $R^2$ score.
   - Dropped the noisy variables to deploy a significantly lighter, faster XGBoost model relying exclusively on the 3 most critical features: **Distance**, **Driver Rating**, and **Driver Age**.

4. **Autonomous Cyberpunk Dashboard:**
   An interactive web application built with Streamlit and Plotly. It features a completely custom, sleek "Dark Neon / Cyberpunk" UI theme using pure CSS and injected Plotly configurations. It includes a raw data viewer, hypothesis test results, model evaluation metrics, and a live AI prediction form.

## 💻 Tech Stack

- **Data Processing:** `pandas`, `numpy`, `regex`
- **Machine Learning:** `scikit-learn`, `xgboost`, `joblib`
- **Statistics:** `scipy.stats`
- **Frontend / UI:** `streamlit`, `streamlit-option-menu`, `plotly.express`, `plotly.graph_objects`

## 📊 How to Run

1. **Install Requirements:**
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit plotly streamlit-option-menu scipy joblib
   ```
2. **Launch the Dashboard:**
   ```bash
   python -m streamlit run dashboard/real_app.py
   ```

## 🏆 Key Analytical Findings

1. **Reverse Causality in Driver Ratings:** Initial assumptions suggested higher-rated drivers take longer (due to carefulness). Statistical analysis proved reverse causality—faster delivery times result in higher ratings from customers.
2. **Vehicle Type Irrelevance:** Despite logical assumptions, the type of vehicle (Motorcycle vs Scooter) had virtually zero impact on delivery times in the final machine learning model, allowing for a simpler, more computationally efficient deployment.
