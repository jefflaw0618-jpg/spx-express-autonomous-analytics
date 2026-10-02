import pandas as pd
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
    
    df = df.dropna()
    df.columns = df.columns.str.strip()
    
    df['Distance_km'] = haversine(df['Restaurant_latitude'], df['Restaurant_longitude'], 
                                  df['Delivery_location_latitude'], df['Delivery_location_longitude'])
                                  
    df = df[df['Distance_km'] <= 50]
    df = df[df['Distance_km'] > 0]
    
    df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
    df['Type_of_order'] = df['Type_of_order'].str.strip()
    df['Delivery_person_ID'] = df['Delivery_person_ID'].str.strip()

    parsed = df['Delivery_person_ID'].str.extract(r'([A-Za-z]+)RES(\d+)DEL(\d+)')
    df['City_Code'] = parsed[0]
    df['Restaurant_No'] = parsed[1]
    df['Delivery_Sequence'] = parsed[2]
    df = df.dropna(subset=['City_Code'])

    features_all = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km', 'Type_of_order', 'Type_of_vehicle', 'City_Code']
    target = 'Time_taken(min)'
    
    X = df[features_all].copy()
    y = df[target]
    
    encoders = {}
    for col in ['Type_of_order', 'Type_of_vehicle', 'City_Code']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. PARSIMONY ANALYSIS FIRST
    base_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    base_xgb.fit(X_train_full, y_train)
    importances = base_xgb.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [features_all[i] for i in sorted_indices]
    
    parsimony_results = []
    print("\nRunning Parsimony Analysis on all 6 features...")
    for i in range(1, len(sorted_features) + 1):
        selected = sorted_features[:i]
        m = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        m.fit(X_train_full[selected], y_train)
        y_pred_sub = m.predict(X_test_full[selected])
        parsimony_results.append({
            "num_features": i,
            "added_feature": selected[-1],
            "r2": float(r2_score(y_test, y_pred_sub)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_sub)))
        })
        
    # 2. SELECT PARSIMONIOUS FEATURES (TOP 3)
    parsimonious_features = sorted_features[:3]
    print(f"\nSelected Parsimonious Features: {parsimonious_features}")
    
    X_train = X_train_full[parsimonious_features]
    X_test = X_test_full[parsimonious_features]

    # 3. TRAIN FINAL MODELS ON PARSIMONIOUS FEATURES ONLY
    models = {
        "Naive Baseline (Mean)": None,
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
        if name == "Naive Baseline (Mean)":
            y_pred = np.full(len(y_test), y_train.mean())
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2_Score": round(r2, 4)})
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model

    feature_importances = {}
    for name, model in models.items():
        if model is not None and hasattr(model, 'feature_importances_'):
            imp = dict(zip(parsimonious_features, [round(float(x), 4) for x in model.feature_importances_]))
            feature_importances[name] = imp

    model_dir = os.path.dirname(__file__)
    joblib.dump(best_model_obj, os.path.join(model_dir, 'real_delivery_regressor.pkl'))
    used_encoders = {k: v for k, v in encoders.items() if k in parsimonious_features}
    joblib.dump(used_encoders, os.path.join(model_dir, 'real_label_encoders.pkl'))
    
    comparison = {
        "results": results,
        "best_model": best_model_name,
        "feature_importances": feature_importances,
        "parsimony": parsimony_results,
        "parsimonious_features": parsimonious_features,
        "dropped_features": [f for f in features_all if f not in parsimonious_features]
    }
    with open(os.path.join(model_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    df.to_csv(os.path.join(model_dir, '..', 'data', 'clean_real_data.csv'), index=False)
    print("Done!")

if __name__ == '__main__':
    train_and_evaluate()
