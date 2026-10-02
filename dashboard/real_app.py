import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Set Cyberpunk theme defaults
cyber_colors = ["#00f3ff", "#b026ff", "#ff007f", "#00ff9d", "#ffdd00", "#8a2be2"]
px.defaults.color_discrete_sequence = cyber_colors
pio.templates.default = "plotly_dark"

from scipy import stats
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Express Data Analytics", page_icon="📦", layout="wide")

# Inject Custom CSS for a cyberpunk/modern dark theme (inspired by provided UI)
st.markdown("""
    <style>
    /* Global Font and Background Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp, p, h1, h2, h3, h4, h5, h6, label, li {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #081021 0%, #02040a 90%) !important;
    }
    
    /* Headers with sleek look */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: 1px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #ffffff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Protect Icon Fonts from being overridden */
    i, .bi, .material-icons, [data-testid="stIconMaterial"], [class*="Icon"] {
        font-family: 'bootstrap-icons', 'Material Icons', 'Material Symbols Rounded' !important;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #00f3ff !important;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #a0aabf !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #b026ff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# Data & Model Loaders
# =========================================================================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_real_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'modeling')
    reg_model = joblib.load(os.path.join(model_dir, 'real_delivery_regressor.pkl'))
    encoders = joblib.load(os.path.join(model_dir, 'real_label_encoders.pkl'))
    return reg_model, encoders

@st.cache_data(show_spinner="Loading model comparison metrics...")
def load_comparison():
    path = os.path.join(os.path.dirname(__file__), '..', 'modeling', 'model_comparison.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# =========================================================================
# Professional Sidebar Navigation
# =========================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00f3ff; font-weight: 800; letter-spacing: 1px;'>EXPRESS<br><span style='color:#b026ff'>DATA ANALYTICS</span></h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = option_menu(
        menu_title=None,
        options=["Operational Overview", "Raw Data Viewer", "Model Comparison", "Hypothesis Testing", "AI Delivery Prediction", "Code Showcase"],
        icons=["bar-chart-line-fill", "table", "robot", "bar-chart-steps", "lightning-charge-fill", "code-slash"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#a0aabf", "font-size": "16px"}, 
            "nav-link": {"font-family": "'Inter', sans-serif", "font-size": "14px", "text-align": "left", "margin":"5px 0px", "color": "#a0aabf", "--hover-color": "rgba(255,255,255,0.05)", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "rgba(176, 38, 255, 0.15)", "color": "#ffffff", "icon-color": "#00f3ff", "border-left": "3px solid #00f3ff", "border-radius": "0px"},
        }
    )
    
    st.markdown("---")
    st.markdown("<p style='font-size: 12px; color: gray; text-align: center;'>Built for SPX Express Portfolio</p>", unsafe_allow_html=True)

# Main Title
st.title("📦 SPX Express - Delivery Intelligence")
st.markdown("Enterprise dashboard powered by real-world logistics data (**45,000+ records**).")

df = load_data()
if df.empty:
    st.error("Data not found. Please run the training script first.")
    st.stop()

# =========================================================================
# PAGE 1: Operational Overview
# =========================================================================
if page == "Operational Overview":
    st.header("Overview of Delivery Operations")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Real Orders", f"{len(df):,}")
    avg_duration = df['Time_taken(min)'].mean()
    col2.metric("Avg Delivery Duration", f"{avg_duration:.0f} mins")
    avg_rating = df['Delivery_person_Ratings'].mean()
    col3.metric("Avg Driver Rating", f"{avg_rating:.2f} ⭐")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders by Vehicle")
        fig1 = px.histogram(df, x="Type_of_vehicle", color="Type_of_vehicle", text_auto=True)
        st.plotly_chart(fig1, use_container_width=True, theme=None)
    with col2:
        st.subheader("Distance vs Time Taken")
        sample_df = df.sample(1000, random_state=42) if len(df) > 1000 else df
        fig2 = px.scatter(sample_df, x="Distance_km", y="Time_taken(min)", color="Type_of_vehicle", opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True, theme=None)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df[['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km', 'Time_taken(min)']].describe().round(2))

# =========================================================================
# PAGE 1.5: Cleaned Dataset Viewer
# =========================================================================
elif page == "Raw Data Viewer":
    st.header("🔍 Dataset Viewer (Parsed IDs)")
    st.markdown("Explore the logistics database. We have used Regex to parse the `Delivery_person_ID` into **City_Code**, **Restaurant_No**, and **Delivery_Sequence**.")
    
    # Load cleaned data so user can see the newly generated columns
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Rows:** {df.shape[0]:,}")
        with col2:
            st.info(f"**Total Columns:** {df.shape[1]}")
            
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.error("Data file not found.")

# =========================================================================
# PAGE 2: Model Comparison
# =========================================================================
elif page == "Model Comparison":
    st.header("🤖 Multi-Model Comparison")
    st.markdown("We trained **6 different models** on the same data and evaluated them on a held-out 20% test set. This allows us to objectively determine which algorithm best predicts delivery times.")
    
    comparison = load_comparison()
    if comparison is None:
        st.error("Model comparison results not found. Please run the training script first.")
        st.stop()
    
    results_df = pd.DataFrame(comparison["results"])
    best = comparison["best_model"]
    
    # Highlight the best model
    st.success(f"🏆 **Best Model: {best}** — selected automatically based on highest R² score.")
    
    if "parsimonious_features" in comparison:
        st.info(f"**Principle of Parsimony Applied:** Rather than training on all variables, we ran a Forward Selection analysis (see bottom of page) to find the most mathematically efficient model. "
                f"We intentionally dropped {', '.join(comparison['dropped_features'])} because they contribute virtually nothing to predictive accuracy (< 0.01 R²). "
                f"These models were trained exclusively on the top {len(comparison['parsimonious_features'])} features: **{', '.join(comparison['parsimonious_features'])}**.")
    
    # Results table
    st.subheader("Performance Metrics (Test Set)")
    
    def highlight_best(row):
        is_best = row['Model'] == best
        return ['background-color: #1a472a; font-weight: bold' if is_best else '' for _ in row]
    
    styled = results_df.style.apply(highlight_best, axis=1).format({
        'RMSE': '{:.2f}',
        'MAE': '{:.2f}',
        'R2_Score': '{:.4f}'
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    st.markdown("""
    | Metric | Meaning |
    |--------|---------|
    | **RMSE** | Root Mean Squared Error — penalises large errors more heavily. Lower is better. |
    | **MAE** | Mean Absolute Error — average absolute prediction error in minutes. Lower is better. |
    | **R²** | Proportion of variance explained by the model. Higher is better (1.0 = perfect). |
    """)
    
    st.markdown("---")
    
    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("R² Score Comparison")
        fig_r2 = px.bar(results_df, x="Model", y="R2_Score", color="Model", text="R2_Score")
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(yaxis_title="R² Score", showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True, theme=None)
    with col2:
        st.subheader("RMSE Comparison")
        fig_rmse = px.bar(results_df, x="Model", y="RMSE", color="Model", text="RMSE")
        fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_rmse.update_layout(yaxis_title="RMSE (mins)", showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True, theme=None)
    
    # Feature importance comparison
    st.markdown("---")
    st.subheader("Feature Importance Comparison (Tree-Based Models)")
    
    fi = comparison.get("feature_importances", {})
    if fi:
        fi_rows = []
        for model_name, imp_dict in fi.items():
            for feat, val in imp_dict.items():
                fi_rows.append({"Model": model_name, "Feature": feat, "Importance": val})
        fi_df = pd.DataFrame(fi_rows)
        fig_fi = px.bar(fi_df, x="Feature", y="Importance", color="Model", barmode="group", text="Importance")
        fig_fi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_fi, use_container_width=True, theme=None)

    # Principle of Parsimony
    parsimony = comparison.get("parsimony", [])
    if parsimony:
        st.markdown("---")
        st.subheader("✂️ Principle of Parsimony (Feature Selection)")
        st.markdown("Occam's razor states that if two models perform similarly, the **simpler model** is preferred. By iteratively adding features starting from the most important one (Forward Selection), we can see exactly when adding more variables stops improving the model.")
        
        pars_df = pd.DataFrame(parsimony)
        # Create a string combining the number of features and the last added feature
        pars_df['Step'] = pars_df['num_features'].astype(str) + " (" + pars_df['added_feature'] + ")"
        
        fig_pars = go.Figure()
        fig_pars.add_trace(go.Scatter(x=pars_df['num_features'], y=pars_df['r2'], mode='lines+markers+text', name='R² Score',
                                      text=pars_df['added_feature'], textposition="top center"))
        fig_pars.update_layout(
            xaxis_title="Number of Features Used",
            yaxis_title="R² Score (Test Set)",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        st.plotly_chart(fig_pars, use_container_width=True, theme=None)
        
        best_r2_val = pars_df['r2'].max()
        st.info(f"As shown above, the R² score plateaus around **{best_r2_val:.4f}**. Adding the final few features contributes almost nothing to predictive power, meaning a simpler model could be deployed in production to save computational resources.")

# =========================================================================
# PAGE 3: Hypothesis Testing
# =========================================================================
elif page == "Hypothesis Testing":
    st.header("📊 Hypothesis Testing")
    st.markdown("Use statistical tests to determine whether there are **significant differences** in delivery time across different categorical groups. All tests use **α = 0.05**.")
    
    test_choice = st.selectbox("Select a Hypothesis Test:", [
        "1. Vehicle Type vs Delivery Time (ANOVA)",
        "2. Order Type vs Delivery Time (ANOVA)",
        "3. Vehicle Type vs Driver Age (ANOVA)",
        "4. Vehicle Type vs Driver Rating (ANOVA)",
        "5. Driver Rating vs Delivery Time (Spearman Rank Correlation)",
        "6. Driver Age vs Delivery Time (Spearman Rank Correlation)",
        "7. Distance vs Delivery Time (Spearman Rank Correlation)",
    ])
    
    alpha = 0.05
    
    if test_choice.startswith("1"):
        st.subheader("One-Way ANOVA: Vehicle Type → Delivery Time")
        st.markdown("**H₀:** There is no significant difference in mean delivery time across vehicle types.")
        st.markdown("**H₁:** At least one vehicle type has a significantly different mean delivery time.")
        
        groups = [group['Time_taken(min)'].values for _, group in df.groupby('Type_of_vehicle')]
        f_stat, p_val = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value", f"{p_val:.6f}")
        
        if p_val < alpha:
            st.error(f"**Result: Reject H₀** (p = {p_val:.6f} < {alpha}). There IS a statistically significant difference in delivery time across vehicle types.")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6f} ≥ {alpha}). No significant difference found.")
        
        st.subheader("Distribution by Vehicle Type")
        fig = px.box(df, x="Type_of_vehicle", y="Time_taken(min)", color="Type_of_vehicle", points=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        st.subheader("Group Summary Statistics")
        summary = df.groupby('Type_of_vehicle')['Time_taken(min)'].agg(['count', 'mean', 'std', 'median']).round(2)
        summary.columns = ['Count', 'Mean (min)', 'Std Dev', 'Median (min)']
        st.dataframe(summary, use_container_width=True)
    
    elif test_choice.startswith("2"):
        st.subheader("One-Way ANOVA: Order Type → Delivery Time")
        st.markdown("**H₀:** There is no significant difference in mean delivery time across order types.")
        st.markdown("**H₁:** At least one order type has a significantly different mean delivery time.")
        
        groups = [group['Time_taken(min)'].values for _, group in df.groupby('Type_of_order')]
        f_stat, p_val = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value", f"{p_val:.6f}")
        
        if p_val < alpha:
            st.error(f"**Result: Reject H₀** (p = {p_val:.6f} < {alpha}). There IS a statistically significant difference in delivery time across order types.")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6f} ≥ {alpha}). No significant difference found.")
        
        fig = px.box(df, x="Type_of_order", y="Time_taken(min)", color="Type_of_order", points=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        summary = df.groupby('Type_of_order')['Time_taken(min)'].agg(['count', 'mean', 'std', 'median']).round(2)
        summary.columns = ['Count', 'Mean (min)', 'Std Dev', 'Median (min)']
        st.dataframe(summary, use_container_width=True)
    
    elif test_choice.startswith("3"):
        st.subheader("One-Way ANOVA: Vehicle Type → Driver Age")
        st.markdown("**H₀:** There is no significant difference in mean driver age across vehicle types.")
        st.markdown("**H₁:** At least one vehicle type has a significantly different mean driver age.")
        
        groups = [group['Delivery_person_Age'].values for _, group in df.groupby('Type_of_vehicle')]
        f_stat, p_val = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value", f"{p_val:.6f}")
        
        if p_val < alpha:
            st.error(f"**Result: Reject H₀** (p = {p_val:.6f} < {alpha}). There IS a statistically significant difference in driver age across vehicle types.")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6f} ≥ {alpha}). No significant difference found.")
        
        fig = px.box(df, x="Type_of_vehicle", y="Delivery_person_Age", color="Type_of_vehicle", points=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        summary = df.groupby('Type_of_vehicle')['Delivery_person_Age'].agg(['count', 'mean', 'std', 'median']).round(2)
        summary.columns = ['Count', 'Mean Age', 'Std Dev', 'Median Age']
        st.dataframe(summary, use_container_width=True)

    elif test_choice.startswith("4"):
        st.subheader("One-Way ANOVA: Vehicle Type → Driver Rating")
        st.markdown("**H₀:** There is no significant difference in mean driver rating across vehicle types.")
        st.markdown("**H₁:** At least one vehicle type has a significantly different mean driver rating.")
        
        groups = [group['Delivery_person_Ratings'].values for _, group in df.groupby('Type_of_vehicle')]
        f_stat, p_val = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value", f"{p_val:.6f}")
        
        if p_val < alpha:
            st.error(f"**Result: Reject H₀** (p = {p_val:.6f} < {alpha}). There IS a statistically significant difference in driver rating across vehicle types.")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6f} ≥ {alpha}). No significant difference found.")
        
        fig = px.box(df, x="Type_of_vehicle", y="Delivery_person_Ratings", color="Type_of_vehicle", points=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        summary = df.groupby('Type_of_vehicle')['Delivery_person_Ratings'].agg(['count', 'mean', 'std', 'median']).round(2)
        summary.columns = ['Count', 'Mean Rating', 'Std Dev', 'Median Rating']
        st.dataframe(summary, use_container_width=True)
    
    elif test_choice.startswith("5"):
        st.subheader("Spearman Rank Correlation: Driver Rating → Delivery Time")
        st.markdown("**H₀:** There is no monotonic correlation between driver rating and delivery time (ρ = 0).")
        st.markdown("**H₁:** There is a significant monotonic correlation (ρ ≠ 0).")
        
        r, p_val = stats.spearmanr(df['Delivery_person_Ratings'], df['Time_taken(min)'])
        
        col1, col2 = st.columns(2)
        col1.metric("Spearman ρ", f"{r:.4f}")
        col2.metric("P-Value", f"{p_val:.6e}")
        
        if p_val < alpha:
            direction = "negative" if r < 0 else "positive"
            st.error(f"**Result: Reject H₀** (p = {p_val:.6e} < {alpha}). There IS a statistically significant **{direction}** rank correlation (ρ = {r:.4f}).")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6e} ≥ {alpha}). No significant rank correlation found.")
        
        sample = df.sample(2000, random_state=42) if len(df) > 2000 else df
        fig = px.scatter(sample, x="Delivery_person_Ratings", y="Time_taken(min)", trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    elif test_choice.startswith("6"):
        st.subheader("Spearman Rank Correlation: Driver Age → Delivery Time")
        st.markdown("**H₀:** There is no monotonic correlation between driver age and delivery time (ρ = 0).")
        st.markdown("**H₁:** There is a significant monotonic correlation (ρ ≠ 0).")
        
        r, p_val = stats.spearmanr(df['Delivery_person_Age'], df['Time_taken(min)'])
        
        col1, col2 = st.columns(2)
        col1.metric("Spearman ρ", f"{r:.4f}")
        col2.metric("P-Value", f"{p_val:.6e}")
        
        if p_val < alpha:
            direction = "negative" if r < 0 else "positive"
            st.error(f"**Result: Reject H₀** (p = {p_val:.6e} < {alpha}). There IS a statistically significant **{direction}** rank correlation (ρ = {r:.4f}).")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6e} ≥ {alpha}). No significant rank correlation found.")
        
        sample = df.sample(2000, random_state=42) if len(df) > 2000 else df
        fig = px.scatter(sample, x="Delivery_person_Age", y="Time_taken(min)", trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    elif test_choice.startswith("7"):
        st.subheader("Spearman Rank Correlation: Distance → Delivery Time")
        st.markdown("**H₀:** There is no monotonic correlation between distance and delivery time (ρ = 0).")
        st.markdown("**H₁:** There is a significant monotonic correlation (ρ ≠ 0).")
        
        r, p_val = stats.spearmanr(df['Distance_km'], df['Time_taken(min)'])
        
        col1, col2 = st.columns(2)
        col1.metric("Spearman ρ", f"{r:.4f}")
        col2.metric("P-Value", f"{p_val:.6e}")
        
        if p_val < alpha:
            direction = "negative" if r < 0 else "positive"
            st.error(f"**Result: Reject H₀** (p = {p_val:.6e} < {alpha}). There IS a statistically significant **{direction}** rank correlation (ρ = {r:.4f}).")
        else:
            st.success(f"**Result: Fail to Reject H₀** (p = {p_val:.6e} ≥ {alpha}). No significant rank correlation found.")
        
        sample = df.sample(2000, random_state=42) if len(df) > 2000 else df
        fig = px.scatter(sample, x="Distance_km", y="Time_taken(min)", trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# =========================================================================
# PAGE 4: AI Delivery Prediction
# =========================================================================
elif page == "AI Delivery Prediction":
    st.header("Autonomous Delivery Time Prediction")
    
    comparison = load_comparison()
    best = comparison["best_model"] if comparison else "XGBoost"
    st.markdown(f"Enter delivery parameters to predict the expected delivery duration using the **{best}** model (best performing).")
    
    try:
        reg_model, encoders = load_models()
        
        with st.form("prediction_form"):
            st.info("Based on the Principle of Parsimony, this AI model requires only the 3 most statistically significant features to make accurate predictions.")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Driver Age", min_value=18, max_value=65, value=30)
                distance = st.number_input("Distance (km)", min_value=1.0, max_value=50.0, value=5.0)
            with col2:
                rating = st.slider("Driver Rating", 1.0, 5.0, 4.5, 0.1)
                
            submit = st.form_submit_button("Predict Delivery Time")
            
            if submit:
                # Ensure the columns match the exact parsimonious features array used during training
                input_data = pd.DataFrame({
                    'Delivery_person_Ratings': [rating],
                    'Distance_km': [distance],
                    'Delivery_person_Age': [age]
                })
                    
                prediction = reg_model.predict(input_data)[0]
                st.success(f"**Predicted Delivery Duration:** {prediction:.0f} minutes")
                if prediction > 45:
                    st.warning("⚠️ High risk of delay! Consider assigning a closer driver.")
                    
    except Exception as e:
        st.error(f"Model not loaded. Please run the training script first. Error: {e}")

# =========================================================================
# PAGE 5: Code Showcase
# =========================================================================
elif page == "Code Showcase":
    st.header("💻 Code Showcase")
    st.markdown("Select a section below to view the **full Python source code** used to build each component of this project.")
    
    code_section = st.selectbox("Select Code Part to View:", [
        "1. Data Cleaning & Feature Engineering",
        "2. Multi-Model Training Pipeline (Full Script)",
        "3. Hypothesis Testing (ANOVA & Pearson)",
        "4. Operational Overview & Visualizations",
        "5. AI Prediction Inference Engine",
        "6. Full Streamlit Dashboard (This App)"
    ])
    
    if code_section.startswith("1"):
        st.markdown("### 🔧 Data Cleaning & Geospatial Feature Engineering")
        st.markdown("This code loads the raw CSV, drops null values, computes distance from raw GPS coordinates using the Haversine formula, and removes anomalous data points (e.g. GPS errors that produce 17,000 km distances).")
        st.code('''import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two GPS points on Earth."""
    R = 6371  # Radius of earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

# Load raw data
data_path = 'data/real_delivery_data.csv'
df = pd.read_csv(data_path)

# Data Cleaning
df = df.dropna()
df.columns = df.columns.str.strip()

# Calculate Distance from raw GPS coordinates
df['Distance_km'] = haversine(
    df['Restaurant_latitude'], df['Restaurant_longitude'], 
    df['Delivery_location_latitude'], df['Delivery_location_longitude']
)

# Data Cleaning: Remove GPS anomalies
# Distances > 50km are physically impossible for local food/parcel delivery
# Distances == 0 mean pickup and delivery are the same point (data error)
df = df[df['Distance_km'] <= 50]
df = df[df['Distance_km'] > 0]

# Strip whitespace from categorical columns
df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
df['Type_of_order'] = df['Type_of_order'].str.strip()

print("Cleaned Data Shape:", df.shape)
print(df.head())''', language='python')
                              
    elif code_section.startswith("2"):
        st.markdown("### 🤖 Multi-Model Training Pipeline (Full Script)")
        st.markdown("This is the **complete `train_real_model.py`** script. It trains 6 models (Naive Baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost), evaluates each on RMSE / MAE / R², extracts feature importances, and saves the best model for dashboard inference.")
        st.code('''import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_delivery_data.csv')
    df = pd.read_csv(data_path)
    
    # Data Cleaning
    df = df.dropna()
    df.columns = df.columns.str.strip()
    
    # Calculate Distance
    df['Distance_km'] = haversine(df['Restaurant_latitude'], df['Restaurant_longitude'], 
                                  df['Delivery_location_latitude'], df['Delivery_location_longitude'])
                                  
    # Remove GPS anomalies
    df = df[df['Distance_km'] <= 50]
    df = df[df['Distance_km'] > 0]
    
    df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
    df['Type_of_order'] = df['Type_of_order'].str.strip()

    print("Real Data Loaded successfully. Shape:", df.shape)
    
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km', 'Type_of_order', 'Type_of_vehicle']
    target = 'Time_taken(min)'
    
    X = df[features].copy()
    y = df[target]
    
    # Encoding categorical variables
    encoders = {}
    for col in ['Type_of_order', 'Type_of_vehicle']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =========================================================================
    # Define all models to compare
    # =========================================================================
    models = {
        "Naive Baseline (Mean)": None,  # Special case: always predicts the mean
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    }

    results = []
    best_r2 = -np.inf
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        print(f"\\nTraining: {name}...")
        
        if name == "Naive Baseline (Mean)":
            # Naive model: always predicts the training set mean
            train_mean = y_train.mean()
            y_pred = np.full(len(y_test), train_mean)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2_Score": round(r2, 4),
        })
        
        print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model

    # =========================================================================
    # Feature importance for tree-based models
    # =========================================================================
    feature_importances = {}
    for name, model in models.items():
        if model is not None and hasattr(model, 'feature_importances_'):
            imp = dict(zip(features, [round(float(x), 4) for x in model.feature_importances_]))
            feature_importances[name] = imp

    # =========================================================================
    # Save everything
    # =========================================================================
    model_dir = os.path.dirname(__file__)
    
    # Save the best model for inference in the dashboard
    if best_model_obj is not None:
        joblib.dump(best_model_obj, os.path.join(model_dir, 'real_delivery_regressor.pkl'))
    joblib.dump(encoders, os.path.join(model_dir, 'real_label_encoders.pkl'))
    
    # Save comparison results as JSON for the dashboard
    comparison = {
        "results": results,
        "best_model": best_model_name,
        "feature_importances": feature_importances,
    }
    with open(os.path.join(model_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save clean data
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_real_data.csv'), index=False)
    
    print(f"\\nBEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print("All models, comparison results, and cleaned data saved!")

if __name__ == "__main__":
    train_and_evaluate()''', language='python')

    elif code_section.startswith("3"):
        st.markdown("### 📊 Hypothesis Testing (ANOVA & Pearson Correlation)")
        st.markdown("This code runs One-Way ANOVA for categorical group comparisons (e.g. Vehicle Type → Delivery Time) and Pearson correlation for continuous variable relationships (e.g. Distance → Delivery Time). Each test reports H₀, H₁, test statistic, p-value, and a reject/fail-to-reject conclusion at α = 0.05.")
        st.code('''from scipy import stats
import plotly.express as px

alpha = 0.05

# ==========================================================
# ANOVA TEST: Vehicle Type vs Delivery Time
# ==========================================================
# H₀: There is no significant difference in mean delivery time across vehicle types.
# H₁: At least one vehicle type has a significantly different mean delivery time.

groups = [group['Time_taken(min)'].values for _, group in df.groupby('Type_of_vehicle')]
f_stat, p_val = stats.f_oneway(*groups)

print(f"F-Statistic: {f_stat:.4f}")
print(f"P-Value: {p_val:.6f}")

if p_val < alpha:
    print(f"Result: Reject H₀ (p = {p_val:.6f} < {alpha})")
    print("There IS a statistically significant difference in delivery time across vehicle types.")
else:
    print(f"Result: Fail to Reject H₀ (p = {p_val:.6f} >= {alpha})")
    print("No significant difference found.")

# Box plot visualization
fig = px.box(df, x="Type_of_vehicle", y="Time_taken(min)", color="Type_of_vehicle", points=False)
fig.show()

# Group summary statistics
summary = df.groupby('Type_of_vehicle')['Time_taken(min)'].agg(['count', 'mean', 'std', 'median']).round(2)
print(summary)


# ==========================================================
# ANOVA TEST: Order Type vs Delivery Time
# ==========================================================
# H₀: There is no significant difference in mean delivery time across order types.
# H₁: At least one order type has a significantly different mean delivery time.

groups = [group['Time_taken(min)'].values for _, group in df.groupby('Type_of_order')]
f_stat, p_val = stats.f_oneway(*groups)

print(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.6f}")


# ==========================================================
# ANOVA TEST: Vehicle Type vs Driver Age
# ==========================================================
# H₀: There is no significant difference in mean driver age across vehicle types.
# H₁: At least one vehicle type has a significantly different mean driver age.

groups = [group['Delivery_person_Age'].values for _, group in df.groupby('Type_of_vehicle')]
f_stat, p_val = stats.f_oneway(*groups)

print(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.6f}")


# ==========================================================
# ANOVA TEST: Vehicle Type vs Driver Rating
# ==========================================================
# H₀: There is no significant difference in mean driver rating across vehicle types.
# H₁: At least one vehicle type has a significantly different mean driver rating.

groups = [group['Delivery_person_Ratings'].values for _, group in df.groupby('Type_of_vehicle')]
f_stat, p_val = stats.f_oneway(*groups)

print(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.6f}")


# ==========================================================
# PEARSON CORRELATION: Driver Rating vs Delivery Time
# ==========================================================
# H₀: There is no linear correlation between driver rating and delivery time (ρ = 0).
# H₁: There is a significant linear correlation (ρ ≠ 0).

r, p_val = stats.pearsonr(df['Delivery_person_Ratings'], df['Time_taken(min)'])

print(f"Pearson r: {r:.4f}")
print(f"P-Value: {p_val:.6e}")

direction = "negative" if r < 0 else "positive"
if p_val < alpha:
    print(f"Reject H₀: Significant {direction} correlation (r = {r:.4f})")

# Scatter plot with OLS trendline
fig = px.scatter(df.sample(2000), x="Delivery_person_Ratings", y="Time_taken(min)", trendline="ols", opacity=0.5)
fig.show()


# ==========================================================
# PEARSON CORRELATION: Driver Age vs Delivery Time
# ==========================================================
r, p_val = stats.pearsonr(df['Delivery_person_Age'], df['Time_taken(min)'])
print(f"Pearson r: {r:.4f}, P-Value: {p_val:.6e}")


# ==========================================================
# PEARSON CORRELATION: Distance vs Delivery Time
# ==========================================================
r, p_val = stats.pearsonr(df['Distance_km'], df['Time_taken(min)'])
print(f"Pearson r: {r:.4f}, P-Value: {p_val:.6e}")''', language='python')

    elif code_section.startswith("4"):
        st.markdown("### 📈 Operational Overview & Visualizations")
        st.markdown("This code computes KPI metrics, builds interactive Plotly charts (histograms, scatter plots, bar charts, box plots), and renders descriptive statistics tables.")
        st.code('''import streamlit as st
import pandas as pd
import plotly.express as px

# Load cleaned data
df = pd.read_csv('data/clean_real_data.csv')

# ==========================================================
# KPI Metrics
# ==========================================================
total_orders = len(df)
avg_duration = df['Time_taken(min)'].mean()
avg_rating = df['Delivery_person_Ratings'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Real Orders", f"{total_orders:,}")
col2.metric("Avg Delivery Duration", f"{avg_duration:.0f} mins")
col3.metric("Avg Driver Rating", f"{avg_rating:.2f} ⭐")

# ==========================================================
# Histogram: Orders by Vehicle
# ==========================================================
fig1 = px.histogram(df, x="Type_of_vehicle", color="Type_of_vehicle", text_auto=True)
st.plotly_chart(fig1, use_container_width=True, theme=None)

# ==========================================================
# Scatter Plot: Distance vs Time Taken (colored by vehicle)
# ==========================================================
sample_df = df.sample(1000, random_state=42) if len(df) > 1000 else df
fig2 = px.scatter(sample_df, x="Distance_km", y="Time_taken(min)", color="Type_of_vehicle", opacity=0.7)
st.plotly_chart(fig2, use_container_width=True, theme=None)

# ==========================================================
# Descriptive Statistics Table
# ==========================================================
desc_stats = df[['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km', 'Time_taken(min)']].describe().round(2)
st.dataframe(desc_stats)

# ==========================================================
# Model Comparison Bar Charts
# ==========================================================
import json

with open('modeling/model_comparison.json', 'r') as f:
    comparison = json.load(f)

results_df = pd.DataFrame(comparison["results"])

# R² Score comparison bar chart
fig_r2 = px.bar(results_df, x="Model", y="R2_Score", color="Model", text="R2_Score")
fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_r2.update_layout(yaxis_title="R² Score", showlegend=False)
st.plotly_chart(fig_r2, use_container_width=True, theme=None)

# RMSE comparison bar chart
fig_rmse = px.bar(results_df, x="Model", y="RMSE", color="Model", text="RMSE")
fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_rmse.update_layout(yaxis_title="RMSE (mins)", showlegend=False)
st.plotly_chart(fig_rmse, use_container_width=True, theme=None)

# Feature Importance grouped bar chart
fi = comparison.get("feature_importances", {})
fi_rows = []
for model_name, imp_dict in fi.items():
    for feat, val in imp_dict.items():
        fi_rows.append({"Model": model_name, "Feature": feat, "Importance": val})
fi_df = pd.DataFrame(fi_rows)

fig_fi = px.bar(fi_df, x="Feature", y="Importance", color="Model", barmode="group", text="Importance")
fig_fi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
st.plotly_chart(fig_fi, use_container_width=True, theme=None)''', language='python')

    elif code_section.startswith("5"):
        st.markdown("### 🚀 AI Prediction Inference Engine")
        st.markdown("This code loads the best saved model (XGBoost), accepts user input through Streamlit form widgets, encodes categorical variables, runs the prediction, and displays the result with a delay risk warning.")
        st.code('''import streamlit as st
import pandas as pd
import joblib

# Load pre-trained best model and label encoders
reg_model = joblib.load('modeling/real_delivery_regressor.pkl')
encoders = joblib.load('modeling/real_label_encoders.pkl')

st.header("Autonomous Delivery Time Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Driver Age", min_value=18, max_value=65, value=30)
        rating = st.slider("Driver Rating", 1.0, 5.0, 4.5, 0.1)
        distance = st.number_input("Distance (km)", min_value=1.0, max_value=50.0, value=5.0)
    with col2:
        order_type = st.selectbox("Order Type", encoders['Type_of_order'].classes_)
        vehicle = st.selectbox("Vehicle Type", encoders['Type_of_vehicle'].classes_)
        
    submit = st.form_submit_button("Predict Delivery Time")
    
    if submit:
        # Prepare input DataFrame matching the training feature order
        input_data = pd.DataFrame({
            'Delivery_person_Age': [age],
            'Delivery_person_Ratings': [rating],
            'Distance_km': [distance],
            'Type_of_order': [order_type],
            'Type_of_vehicle': [vehicle]
        })
        
        # Encode categorical variables using the same LabelEncoders from training
        input_data['Type_of_order'] = encoders['Type_of_order'].transform(input_data['Type_of_order'])
        input_data['Type_of_vehicle'] = encoders['Type_of_vehicle'].transform(input_data['Type_of_vehicle'])
            
        # Run inference
        prediction = reg_model.predict(input_data)[0]
        
        st.success(f"**Predicted Delivery Duration:** {prediction:.0f} minutes")
        
        # Business logic: flag high-risk deliveries
        if prediction > 45:
            st.warning("⚠️ High risk of delay! Consider assigning a closer driver.")''', language='python')

    elif code_section.startswith("6"):
        st.markdown("### 🏗️ Full Streamlit Dashboard Architecture")
        st.markdown("This is the **complete `real_app.py`** source code that powers this entire dashboard. It includes all 5 pages: Operational Overview, Model Comparison, Hypothesis Testing, AI Prediction, and this Code Showcase.")
        
        # Read the actual file dynamically
        app_path = os.path.abspath(__file__)
        try:
            with open(app_path, 'r', encoding='utf-8') as f:
                full_code = f.read()
            st.code(full_code, language='python')
        except Exception as e:
            st.error(f"Could not read source file: {e}")

