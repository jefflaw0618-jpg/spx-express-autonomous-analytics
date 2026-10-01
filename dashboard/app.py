import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="SPX Express Autonomous Dashboard", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spx_delivery_data.csv')
    return pd.read_csv(data_path)

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'modeling')
    reg_model = joblib.load(os.path.join(model_dir, 'delivery_time_regressor.pkl'))
    encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    return reg_model, encoders

st.title("📦 SPX Express - Autonomous Analytics & Prediction Dashboard")
st.markdown("Monitor delivery operations across major Malaysian Hubs and predict delivery times using AI.")

df = load_data()

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Operational Overview", "Hub Performance Analysis", "AI Delivery Prediction"])

if page == "Operational Overview":
    st.header("Overview of Delivery Operations")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders Processed", f"{len(df):,}")
    
    delayed_count = len(df[df['Delivery_Status'] == 'Delayed'])
    delay_rate = (delayed_count / len(df)) * 100
    col2.metric("Overall Delay Rate", f"{delay_rate:.2f}%")
    
    avg_duration = df['Delivery_Duration_mins'].mean()
    col3.metric("Avg Delivery Duration", f"{avg_duration:.0f} mins")
    
    avg_rating = df['Driver_Rating'].mean()
    col4.metric("Avg Driver Rating", f"{avg_rating:.2f} ⭐")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders Volume by Hub")
        fig1 = px.histogram(df, x="Hub_Location", color="Hub_Location", text_auto=True)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Delivery Status Distribution")
        fig2 = px.pie(df, names="Delivery_Status", color="Delivery_Status", 
                      color_discrete_map={'On-Time':'green', 'Delayed':'red'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Recent Order Tracking Data")
    st.dataframe(df.tail(10))

elif page == "Hub Performance Analysis":
    st.header("Hub Performance Analysis")
    
    selected_hub = st.selectbox("Select Hub", df['Hub_Location'].unique())
    hub_data = df[df['Hub_Location'] == selected_hub]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Delay Rate by Weather in {selected_hub}")
        delay_by_weather = hub_data.groupby(['Weather_Conditions', 'Delivery_Status']).size().reset_index(name='Count')
        fig = px.bar(delay_by_weather, x="Weather_Conditions", y="Count", color="Delivery_Status", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader(f"Avg Delivery Time by Vehicle in {selected_hub}")
        avg_time_vehicle = hub_data.groupby('Vehicle_Type')['Delivery_Duration_mins'].mean().reset_index()
        fig2 = px.bar(avg_time_vehicle, x="Vehicle_Type", y="Delivery_Duration_mins", color="Vehicle_Type")
        st.plotly_chart(fig2, use_container_width=True)
        
elif page == "AI Delivery Prediction":
    st.header("Autonomous Delivery Time Prediction")
    st.markdown("Enter delivery parameters to predict the expected delivery duration using our Random Forest Model.")
    
    try:
        reg_model, encoders = load_models()
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                hub = st.selectbox("Origin Hub", encoders['Hub_Location'].classes_)
                vehicle = st.selectbox("Vehicle Type", encoders['Vehicle_Type'].classes_)
                distance = st.number_input("Distance (km)", min_value=1.0, max_value=200.0, value=15.0)
                
            with col2:
                weather = st.selectbox("Weather Conditions", encoders['Weather_Conditions'].classes_)
                traffic = st.selectbox("Traffic Level", encoders['Traffic_Level'].classes_)
                rating = st.slider("Driver Rating", 1.0, 5.0, 4.5, 0.1)
                
            submit = st.form_submit_button("Predict Delivery Time")
            
            if submit:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Hub_Location': [hub],
                    'Vehicle_Type': [vehicle],
                    'Distance_km': [distance],
                    'Weather_Conditions': [weather],
                    'Traffic_Level': [traffic],
                    'Driver_Rating': [rating]
                })
                
                # Encode input
                for col in ['Hub_Location', 'Vehicle_Type', 'Weather_Conditions', 'Traffic_Level']:
                    input_data[col] = encoders[col].transform(input_data[col])
                    
                prediction = reg_model.predict(input_data)[0]
                
                st.success(f"**Predicted Delivery Duration:** {prediction:.0f} minutes")
                
                if prediction > 120:
                    st.warning("High risk of delay! Consider re-routing or assigning a different vehicle type.")
                    
    except Exception as e:
        st.error(f"Model not loaded or trained yet. Please run the modeling script first. Error: {e}")
