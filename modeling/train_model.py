import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spx_delivery_data.csv')
    df = pd.read_csv(data_path)

    print("Data Loaded successfully. Shape:", df.shape)
    
    # Feature Engineering
    # Convert time strings to datetime if needed
    
    # Selecting Features
    features = ['Hub_Location', 'Vehicle_Type', 'Distance_km', 'Weather_Conditions', 'Traffic_Level', 'Driver_Rating']
    target_regression = 'Delivery_Duration_mins'
    target_classification = 'Delivery_Status'
    
    X = df[features].copy()
    y_reg = df[target_regression]
    
    # Encoding categorical variables
    encoders = {}
    for col in ['Hub_Location', 'Vehicle_Type', 'Weather_Conditions', 'Traffic_Level']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    y_class = LabelEncoder().fit_transform(df[target_classification])

    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # 1. Regression Model (Predicting Delivery Duration)
    print("\nTraining Regression Model (Random Forest)...")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    
    y_pred_reg = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"Regression RMSE: {rmse:.2f} mins")
    print(f"Regression R2 Score: {r2:.2f}")

    # 2. Classification Model (Predicting Delayed vs On-Time)
    print("\nTraining Classification Model (Random Forest)...")
    cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cls_model.fit(X_train, y_train_cls)
    
    y_pred_cls = cls_model.predict(X_test)
    acc = accuracy_score(y_test_cls, y_pred_cls)
    print(f"Classification Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls, target_names=['Delayed', 'On-Time']))
    
    # Feature Importance
    importances = reg_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (Regression):\n", feat_imp)

    # Save models and encoders
    model_dir = os.path.dirname(__file__)
    joblib.dump(reg_model, os.path.join(model_dir, 'delivery_time_regressor.pkl'))
    joblib.dump(cls_model, os.path.join(model_dir, 'delivery_status_classifier.pkl'))
    joblib.dump(encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    
    print("\nModels and encoders saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()
