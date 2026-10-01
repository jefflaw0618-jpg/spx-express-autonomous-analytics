# 📦 SPX Express - Autonomous Analytics & Prediction Project

This project was built as part of the preparation and portfolio development for the **Data Analytics Intern - SPX Express** role. It demonstrates end-to-end data generation, predictive modeling, and the deployment of an autonomous dashboard to track and predict delivery operations across Malaysia.

## 🚀 Project Overview

The core objective of this project is to identify operational gaps in delivery operations and provide data-driven tools to mitigate delays.

1. **Synthetic Data Generation (`data/generate_data.py`):**
   Generates a highly realistic synthetic logistics dataset simulating 10,000 delivery orders across major Malaysian Hubs (Kuala Lumpur, Penang, Johor Bahru, etc.). Features include distance, weather, traffic conditions, driver rating, and vehicle type.

2. **Autonomous Delivery Prediction Model (`modeling/train_model.py`):**
   An AI-based predictive modeling pipeline using `scikit-learn` (Random Forest) to predict:
   - **Delivery Duration (Regression)**: How long will a delivery take based on live conditions?
   - **Delivery Status (Classification)**: Will the delivery be On-Time or Delayed?

3. **Autonomous Dashboard (`dashboard/app.py`):**
   An interactive web application built with Streamlit and Plotly. It monitors operational performance, tracks delay rates, and allows dispatchers to input parameters (like weather and traffic) to predict expected delivery times autonomously.

## 🛠️ Tech Stack
- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Joblib
- **Visualization & Dashboard**: Streamlit, Plotly

## 📂 Project Structure
```
spx_analytics_project/
│
├── data/
│   ├── generate_data.py          # Script to generate synthetic dataset
│   └── spx_delivery_data.csv     # Generated dataset (10,000 rows)
│
├── modeling/
│   ├── train_model.py            # Model training & evaluation script
│   ├── delivery_time_regressor.pkl # Saved Regression Model
│   ├── delivery_status_classifier.pkl # Saved Classification Model
│   └── label_encoders.pkl        # Saved Encoders for Inference
│
├── dashboard/
│   └── app.py                    # Streamlit Dashboard App
│
└── README.md                     # Project Documentation
```

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn joblib streamlit plotly
   ```

2. **Generate the Data (Optional):**
   ```bash
   python data/generate_data.py
   ```

3. **Train the Models:**
   ```bash
   python modeling/train_model.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## 📈 Key Insights & Operational Value
- By predicting delays proactively, hub managers can reroute packages or allocate different vehicle types (e.g., Truck vs Van) depending on weather and traffic bottlenecks.
- The dashboard highlights hub-specific performance, allowing senior management to identify which hubs require resource optimization.
