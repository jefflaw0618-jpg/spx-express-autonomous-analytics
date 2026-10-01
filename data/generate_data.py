import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_spx_delivery_data(num_records=10000):
    np.random.seed(42)
    random.seed(42)
    
    hubs = ['Kuala Lumpur', 'Penang', 'Johor Bahru', 'Selangor', 'Melaka']
    vehicle_types = ['Motorcycle', 'Van', 'Truck']
    weather_cond = ['Clear', 'Cloudy', 'Rain', 'Heavy Rain']
    traffic_levels = ['Low', 'Medium', 'High', 'Jam']
    
    data = []
    
    start_date = datetime(2026, 9, 1)
    
    for i in range(num_records):
        order_id = f"SPX{1000000 + i}"
        hub = random.choice(hubs)
        vehicle = random.choice(vehicle_types)
        weather = np.random.choice(weather_cond, p=[0.5, 0.3, 0.15, 0.05])
        traffic = np.random.choice(traffic_levels, p=[0.3, 0.4, 0.2, 0.1])
        
        # Distance based on vehicle
        if vehicle == 'Motorcycle':
            distance = round(random.uniform(1.0, 15.0), 2)
        elif vehicle == 'Van':
            distance = round(random.uniform(5.0, 50.0), 2)
        else:
            distance = round(random.uniform(20.0, 150.0), 2)
            
        driver_rating = round(random.uniform(3.5, 5.0), 1)
        
        # Time generation
        order_time = start_date + timedelta(days=random.randint(0, 30), hours=random.randint(8, 20), minutes=random.randint(0, 59))
        
        # Dispatch time (usually 30 mins to 4 hours after order)
        dispatch_delay = timedelta(minutes=random.randint(30, 240))
        dispatch_time = order_time + dispatch_delay
        
        # Calculate delivery time based on distance, traffic, weather
        base_time = distance * 2 # 2 mins per km base
        
        weather_multiplier = {'Clear': 1.0, 'Cloudy': 1.1, 'Rain': 1.5, 'Heavy Rain': 2.0}
        traffic_multiplier = {'Low': 1.0, 'Medium': 1.2, 'High': 1.8, 'Jam': 2.5}
        
        actual_time_mins = base_time * weather_multiplier[weather] * traffic_multiplier[traffic]
        actual_time_mins += random.uniform(-10, 20) # Add some noise
        
        delivery_time = dispatch_time + timedelta(minutes=max(10, actual_time_mins))
        
        # Expected delivery time is just simple heuristic for Shopee
        expected_time_mins = base_time * 1.2 # Baseline buffer
        expected_delivery = dispatch_time + timedelta(minutes=max(15, expected_time_mins))
        
        # Status
        status = 'On-Time' if delivery_time <= expected_delivery else 'Delayed'
        
        # Extract durations for ML
        dispatch_to_delivery_mins = round((delivery_time - dispatch_time).total_seconds() / 60, 2)
        
        data.append([
            order_id, hub, vehicle, distance, weather, traffic, driver_rating, 
            order_time.strftime('%Y-%m-%d %H:%M:%S'), 
            dispatch_time.strftime('%Y-%m-%d %H:%M:%S'), 
            delivery_time.strftime('%Y-%m-%d %H:%M:%S'),
            dispatch_to_delivery_mins,
            status
        ])
        
    df = pd.DataFrame(data, columns=[
        'Order_ID', 'Hub_Location', 'Vehicle_Type', 'Distance_km', 'Weather_Conditions', 
        'Traffic_Level', 'Driver_Rating', 'Order_Time', 'Dispatch_Time', 'Delivery_Time',
        'Delivery_Duration_mins', 'Delivery_Status'
    ])
    
    output_path = os.path.join(os.path.dirname(__file__), 'spx_delivery_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {num_records} records to {output_path}")

if __name__ == "__main__":
    generate_spx_delivery_data()
