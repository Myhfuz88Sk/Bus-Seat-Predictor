# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # ML Libraries
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib

# # Flask for API
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# class ImprovedRedBusDemandForecaster:
#     def __init__(self):
#         self.models = {}
#         self.scalers = {}
#         self.encoders = {}
#         self.feature_names = []
        
#         # City mapping with tier classification
#         self.cities = {
#             '01': {'name': 'Mumbai', 'tier': 1, 'population': 12.5},
#             '02': {'name': 'Delhi', 'tier': 1, 'population': 11.0},
#             '03': {'name': 'Bangalore', 'tier': 1, 'population': 8.5},
#             '04': {'name': 'Chennai', 'tier': 1, 'population': 7.0},
#             '05': {'name': 'Hyderabad', 'tier': 1, 'population': 6.8},
#             '06': {'name': 'Pune', 'tier': 1, 'population': 3.1},
#             '07': {'name': 'Kolkata', 'tier': 1, 'population': 4.5},
#             '08': {'name': 'Ahmedabad', 'tier': 2, 'population': 5.6},
#             '09': {'name': 'Jaipur', 'tier': 2, 'population': 3.1},
#             '10': {'name': 'Kochi', 'tier': 2, 'population': 0.6},
#             '11': {'name': 'Coimbatore', 'tier': 2, 'population': 1.1},
#             '12': {'name': 'Indore', 'tier': 2, 'population': 1.9},
#             '13': {'name': 'Nagpur', 'tier': 2, 'population': 2.4},
#             '14': {'name': 'Lucknow', 'tier': 2, 'population': 2.8},
#             '15': {'name': 'Bhopal', 'tier': 2, 'population': 1.8},
#             '16': {'name': 'Visakhapatnam', 'tier': 3, 'population': 1.7},
#             '17': {'name': 'Surat', 'tier': 2, 'population': 4.5},
#             '18': {'name': 'Vadodara', 'tier': 3, 'population': 1.7},
#             '19': {'name': 'Thiruvananthapuram', 'tier': 3, 'population': 0.75},
#             '20': {'name': 'Guwahati', 'tier': 3, 'population': 0.96}
#         }
        
#         # Enhanced operator data
#         self.operators = {
#             'redbus': {'market_share': 0.35, 'price_factor': 1.0},
#             'abhibus': {'market_share': 0.15, 'price_factor': 0.95},
#             'makemytrip': {'market_share': 0.12, 'price_factor': 1.05},
#             'yatra': {'market_share': 0.08, 'price_factor': 0.98},
#             'goibibo': {'market_share': 0.10, 'price_factor': 1.02},
#             'cleartrip': {'market_share': 0.06, 'price_factor': 1.03},
#             'paytm': {'market_share': 0.07, 'price_factor': 0.97},
#             'ixigo': {'market_share': 0.04, 'price_factor': 0.94},
#             'easemytrip': {'market_share': 0.02, 'price_factor': 0.92},
#             'mobikwik': {'market_share': 0.01, 'price_factor': 0.90}
#         }

#     def generate_enhanced_synthetic_data(self, n_samples=15000):
#         """Generate more realistic synthetic data with better patterns"""
#         np.random.seed(42)
#         data = []
#         start_date = datetime(2022, 1, 1)
        
#         # Pre-calculate distance matrix
#         distance_matrix = self._calculate_distance_matrix()
        
#         for i in range(n_samples):
#             # Random date with bias towards recent dates
#             days_offset = int(np.random.exponential(200))  # Exponential distribution
#             days_offset = min(days_offset, 730)  # Cap at 2 years
#             booking_date = start_date + timedelta(days=days_offset)
            
#             # Select origin and destination with realistic probabilities
#             origin_id, dest_id = self._select_realistic_route()
            
#             # Select operator based on market share
#             operator = self._select_operator_by_market_share()
            
#             # Enhanced features
#             day_of_week = booking_date.weekday()
#             month = booking_date.month
#             day_of_month = booking_date.day
#             is_weekend = 1 if day_of_week >= 5 else 0
#             is_holiday = self._is_enhanced_holiday(booking_date)
            
#             # Distance and route features
#             distance = distance_matrix.get((origin_id, dest_id), 300)
#             route_type = self._get_route_type(origin_id, dest_id)
            
#             # Population and tier factors
#             origin_pop = self.cities[origin_id]['population']
#             dest_pop = self.cities[dest_id]['population']
#             origin_tier = self.cities[origin_id]['tier']
#             dest_tier = self.cities[dest_id]['tier']
            
#             # Operator factors
#             operator_share = self.operators[operator]['market_share']
#             price_factor = self.operators[operator]['price_factor']
            
#             # Advanced demand calculation
#             base_demand = self._calculate_advanced_demand(
#                 origin_pop, dest_pop, origin_tier, dest_tier,
#                 distance, route_type, operator_share, price_factor,
#                 month, day_of_week, is_weekend, is_holiday
#             )
            
#             # Add realistic noise
#             demand = max(5, int(base_demand + np.random.normal(0, base_demand * 0.15)))
            
#             data.append({
#                 'date': booking_date,
#                 'origin_id': origin_id,
#                 'dest_id': dest_id,
#                 'bus_operator': operator,
#                 'day_of_week': day_of_week,
#                 'month': month,
#                 'day_of_month': day_of_month,
#                 'is_weekend': is_weekend,
#                 'is_holiday': is_holiday,
#                 'distance': distance,
#                 'route_type': route_type,
#                 'origin_population': origin_pop,
#                 'dest_population': dest_pop,
#                 'origin_tier': origin_tier,
#                 'dest_tier': dest_tier,
#                 'operator_market_share': operator_share,
#                 'price_factor': price_factor,
#                 'seasonal_factor': self._get_enhanced_seasonal_factor(month, day_of_week),
#                 'demand': demand
#             })
        
#         return pd.DataFrame(data)
    
#     def _calculate_distance_matrix(self):
#         """Calculate realistic distances between cities"""
#         # Simplified distance calculation based on city codes and geographic knowledge
#         distances = {}
#         city_ids = list(self.cities.keys())
        
#         for i, origin in enumerate(city_ids):
#             for j, dest in enumerate(city_ids):
#                 if origin != dest:
#                     # Base distance calculation with some realistic patterns
#                     base_dist = abs(int(origin) - int(dest)) * 150
#                     # Add some randomness for realism
#                     actual_dist = base_dist + np.random.uniform(-50, 100)
#                     distances[(origin, dest)] = max(50, actual_dist)
        
#         return distances
    
#     def _select_realistic_route(self):
#         """Select origin-destination pairs with realistic probabilities"""
#         city_ids = list(self.cities.keys())
#         tier1_cities = [k for k, v in self.cities.items() if v['tier'] == 1]
        
#         # 60% routes involve at least one tier-1 city
#         if np.random.random() < 0.6:
#             origin = np.random.choice(tier1_cities)
#             # Higher probability of tier-1 to tier-1 routes
#             if np.random.random() < 0.4:
#                 dest = np.random.choice([c for c in tier1_cities if c != origin])
#             else:
#                 dest = np.random.choice([c for c in city_ids if c != origin])
#         else:
#             # Random route
#             origin = np.random.choice(city_ids)
#             dest = np.random.choice([c for c in city_ids if c != origin])
        
#         return origin, dest
    
#     def _select_operator_by_market_share(self):
#         """Select operator based on market share"""
#         operators = list(self.operators.keys())
#         weights = [self.operators[op]['market_share'] for op in operators]
#         return np.random.choice(operators, p=weights)
    
#     def _get_route_type(self, origin_id, dest_id):
#         """Classify route type"""
#         origin_tier = self.cities[origin_id]['tier']
#         dest_tier = self.cities[dest_id]['tier']
        
#         if origin_tier == 1 and dest_tier == 1:
#             return 3  # Metro to Metro
#         elif origin_tier <= 2 and dest_tier <= 2:
#             return 2  # Major city routes
#         else:
#             return 1  # Other routes
    
#     def _is_enhanced_holiday(self, date):
#         """Enhanced holiday detection including festivals"""
#         holidays = [
#             (1, 26), (3, 8), (8, 15), (10, 2), (12, 25),  # National holidays
#             (1, 1), (1, 14), (4, 14), (5, 1), (11, 14)    # Regional festivals
#         ]
        
#         # Festival seasons (approximate)
#         festival_months = {10, 11, 4, 3}  # Diwali, Dussehra, Holi seasons
        
#         is_national_holiday = (date.month, date.day) in holidays
#         is_festival_season = date.month in festival_months
        
#         return 2 if is_national_holiday else (1 if is_festival_season else 0)
    
#     def _calculate_advanced_demand(self, origin_pop, dest_pop, origin_tier, dest_tier,
#                                    distance, route_type, operator_share, price_factor,
#                                    month, day_of_week, is_weekend, is_holiday):
#         """Advanced demand calculation with multiple factors"""
        
#         # Population factor (log scale for better distribution)
#         pop_factor = np.log(origin_pop + dest_pop + 1) * 5
        
#         # Tier factor
#         tier_factor = (5 - origin_tier) + (5 - dest_tier)  # Lower tier = higher factor
        
#         # Distance factor (optimal distance around 300-500km)
#         if distance < 100:
#             distance_factor = 0.5  # Too short for bus preference
#         elif distance < 500:
#             distance_factor = 1.0 + (500 - distance) / 1000  # Sweet spot
#         else:
#             distance_factor = 0.8 - (distance - 500) / 2000  # Too long
        
#         distance_factor = max(0.2, distance_factor)
        
#         # Route type factor
#         route_factor = route_type * 0.5
        
#         # Operator factor
#         operator_factor = operator_share * 100 * (2 - price_factor)  # Lower price = higher demand
        
#         # Temporal factors
#         weekend_factor = 1.5 if is_weekend else 1.0
#         holiday_factor = 1.0 + is_holiday * 0.3
        
#         # Seasonal pattern
#         seasonal_factor = self._get_enhanced_seasonal_factor(month, day_of_week)
        
#         # Day of week pattern
#         weekday_factors = {0: 1.0, 1: 1.1, 2: 1.0, 3: 1.1, 4: 1.4, 5: 1.8, 6: 1.6}
#         weekday_factor = weekday_factors.get(day_of_week, 1.0)
        
#         # Combine all factors
#         base_demand = (
#             pop_factor * 0.3 +
#             tier_factor * 0.2 +
#             distance_factor * 0.2 +
#             route_factor * 0.1 +
#             operator_factor * 0.2
#         ) * weekend_factor * holiday_factor * seasonal_factor * weekday_factor
        
#         return base_demand
    
#     def _get_enhanced_seasonal_factor(self, month, day_of_week):
#         """Enhanced seasonal factors"""
#         # Base seasonal pattern
#         seasonal_base = {
#             1: 1.3, 2: 1.1, 3: 0.9, 4: 1.4, 5: 1.5, 6: 1.4,
#             7: 0.8, 8: 0.9, 9: 0.8, 10: 1.2, 11: 1.3, 12: 1.4
#         }
        
#         base_factor = seasonal_base.get(month, 1.0)
        
#         # Weekend boost in peak months
#         if day_of_week >= 5 and month in [4, 5, 6, 10, 11, 12]:
#             base_factor *= 1.2
        
#         return base_factor
    
#     def prepare_enhanced_features(self, df):
#         """Prepare enhanced feature set"""
#         df = df.copy()
        
#         # Encode categorical variables
#         if 'origin_id' not in self.encoders:
#             self.encoders['origin_id'] = LabelEncoder()
#             self.encoders['dest_id'] = LabelEncoder()
#             self.encoders['bus_operator'] = LabelEncoder()
            
#             df['origin_encoded'] = self.encoders['origin_id'].fit_transform(df['origin_id'])
#             df['dest_encoded'] = self.encoders['dest_id'].fit_transform(df['dest_id'])
#             df['operator_encoded'] = self.encoders['bus_operator'].fit_transform(df['bus_operator'])
#         else:
#             df['origin_encoded'] = self.encoders['origin_id'].transform(df['origin_id'])
#             df['dest_encoded'] = self.encoders['dest_id'].transform(df['dest_id'])
#             df['operator_encoded'] = self.encoders['bus_operator'].transform(df['bus_operator'])
        
#         # Create interaction features
#         df['origin_dest_interaction'] = df['origin_encoded'] * df['dest_encoded']
#         df['pop_distance_ratio'] = (df['origin_population'] + df['dest_population']) / (df['distance'] / 100)
#         df['tier_route_score'] = df['origin_tier'] + df['dest_tier'] + df['route_type']
        
#         # Select enhanced features
#         feature_columns = [
#             'origin_encoded', 'dest_encoded', 'operator_encoded',
#             'day_of_week', 'month', 'day_of_month', 'is_weekend', 'is_holiday',
#             'distance', 'route_type', 'origin_population', 'dest_population',
#             'origin_tier', 'dest_tier', 'operator_market_share', 'price_factor',
#             'seasonal_factor', 'origin_dest_interaction', 'pop_distance_ratio',
#             'tier_route_score'
#         ]
        
#         self.feature_names = feature_columns
#         return df[feature_columns]
    
#     def train_enhanced_models(self, df):
#         """Train models with enhanced features and better parameters"""
#         print("Preparing enhanced features...")
#         X = self.prepare_enhanced_features(df)
#         y = df['demand']
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
#         )
        
#         # Scale features
#         self.scalers['scaler'] = StandardScaler()
#         X_train_scaled = self.scalers['scaler'].fit_transform(X_train)
#         X_test_scaled = self.scalers['scaler'].transform(X_test)
        
#         # Enhanced models with better parameters
#         models_to_train = {
#             'random_forest': RandomForestRegressor(
#                 n_estimators=150,
#                 max_depth=15,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=42,
#                 n_jobs=-1
#             ),
#             'gradient_boosting': GradientBoostingRegressor(
#                 n_estimators=150,
#                 learning_rate=0.1,
#                 max_depth=8,
#                 random_state=42
#             ),
#             'linear_regression': LinearRegression()
#         }
        
#         print("Training enhanced models...")
#         results = {}
        
#         for name, model in models_to_train.items():
#             print(f"Training {name}...")
            
#             if name == 'linear_regression':
#                 # Add polynomial features for linear regression
#                 poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#                 X_train_poly = poly_features.fit_transform(X_train_scaled)
#                 X_test_poly = poly_features.transform(X_test_scaled)
                
#                 model.fit(X_train_poly, y_train)
#                 y_pred = model.predict(X_test_poly)
                
#                 # Store polynomial transformer
#                 self.scalers['poly_features'] = poly_features
#             else:
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
            
#             # Evaluate
#             mae = mean_absolute_error(y_test, y_pred)
#             mse = mean_squared_error(y_test, y_pred)
#             rmse = np.sqrt(mse)
#             r2 = r2_score(y_test, y_pred)
            
#             # Cross-validation score
#             if name != 'linear_regression':
#                 cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
#                 cv_mean = cv_scores.mean()
#             else:
#                 cv_mean = r2  # Skip CV for polynomial linear regression
            
#             results[name] = {
#                 'model': model,
#                 'mae': mae,
#                 'rmse': rmse,
#                 'r2': r2,
#                 'cv_r2': cv_mean,
#                 'predictions': y_pred
#             }
            
#             print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, CV-R2: {cv_mean:.3f}")
        
#         # Select best model based on cross-validation R2 score
#         best_model_name = max(results.keys(), key=lambda k: results[k]['cv_r2'])
#         self.models['best'] = results[best_model_name]['model']
#         self.models['best_name'] = best_model_name
        
#         print(f"\nBest model: {best_model_name} with CV-R2: {results[best_model_name]['cv_r2']:.3f}")
        
#         return results
    
#     def predict_demand(self, origin_id, dest_id, bus_operator, date_str):
#         """Enhanced prediction with better feature engineering"""
#         try:
#             date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
#             # Calculate distance
#             distance = abs(int(origin_id) - int(dest_id)) * 150 + np.random.uniform(-50, 50)
#             distance = max(50, distance)
            
#             # Get city information
#             origin_info = self.cities.get(origin_id, {'tier': 2, 'population': 1.0})
#             dest_info = self.cities.get(dest_id, {'tier': 2, 'population': 1.0})
#             operator_info = self.operators.get(bus_operator, {'market_share': 0.05, 'price_factor': 1.0})
            
#             # Create enhanced features
#             features = {
#                 'origin_id': origin_id,
#                 'dest_id': dest_id,
#                 'bus_operator': bus_operator,
#                 'day_of_week': date_obj.weekday(),
#                 'month': date_obj.month,
#                 'day_of_month': date_obj.day,
#                 'is_weekend': 1 if date_obj.weekday() >= 5 else 0,
#                 'is_holiday': self._is_enhanced_holiday(date_obj),
#                 'distance': distance,
#                 'route_type': self._get_route_type(origin_id, dest_id),
#                 'origin_population': origin_info['population'],
#                 'dest_population': dest_info['population'],
#                 'origin_tier': origin_info['tier'],
#                 'dest_tier': dest_info['tier'],
#                 'operator_market_share': operator_info['market_share'],
#                 'price_factor': operator_info['price_factor'],
#                 'seasonal_factor': self._get_enhanced_seasonal_factor(date_obj.month, date_obj.weekday())
#             }
            
#             # Create DataFrame and prepare features
#             feature_df = pd.DataFrame([features])
#             X = self.prepare_enhanced_features(feature_df)
            
#             # Make prediction
#             if self.models['best_name'] == 'linear_regression':
#                 X_scaled = self.scalers['scaler'].transform(X)
#                 X_poly = self.scalers['poly_features'].transform(X_scaled)
#                 prediction = self.models['best'].predict(X_poly)[0]
#             else:
#                 prediction = self.models['best'].predict(X)[0]
            
#             return max(10, int(round(prediction)))
        
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             return 50
    
#     def save_model(self, filepath='enhanced_redbus_model.pkl'):
#         """Save enhanced model"""
#         model_data = {
#             'models': self.models,
#             'scalers': self.scalers,
#             'encoders': self.encoders,
#             'feature_names': self.feature_names,
#             'cities': self.cities,
#             'operators': self.operators
#         }
#         joblib.dump(model_data, filepath)
#         print(f"Enhanced model saved to {filepath}")
    
#     def load_model(self, filepath='enhanced_redbus_model.pkl'):
#         """Load enhanced model"""
#         model_data = joblib.load(filepath)
#         self.models = model_data['models']
#         self.scalers = model_data['scalers']
#         self.encoders = model_data['encoders']
#         self.feature_names = model_data['feature_names']
#         self.cities = model_data['cities']
#         self.operators = model_data['operators']
#         print(f"Enhanced model loaded from {filepath}")

# # Flask API with CORS enabled
# app = Flask(__name__)
# CORS(app)

# # Global forecaster instance
# forecaster = ImprovedRedBusDemandForecaster()

# @app.route('/')
# def home():
#     return jsonify({
#         'message': 'RedBus Demand Forecasting API',
#         'version': '2.0 Enhanced',
#         'endpoints': ['/forecast', '/retrain', '/health']
#     })

# @app.route('/health')
# def health():
#     return jsonify({'status': 'healthy', 'model_loaded': bool(forecaster.models)})

# @app.route('/forecast', methods=['POST'])
# def forecast():
#     try:
#         data = request.get_json()
#         print(f"Received forecast request: {data}")
        
#         origin_id = data.get('originid')
#         dest_id = data.get('destid')
#         bus_operator = data.get('busoperator', 'redbus')
#         date_str = data.get('doj')
        
#         if not all([origin_id, dest_id, date_str]):
#             return jsonify({'success': False, 'error': 'Missing required fields'})
        
#         # Predict demand
#         demand = forecaster.predict_demand(origin_id, dest_id, bus_operator, date_str)
        
#         origin_name = forecaster.cities.get(origin_id, {}).get('name', 'Unknown')
#         dest_name = forecaster.cities.get(dest_id, {}).get('name', 'Unknown')
        
#         return jsonify({
#             'success': True,
#             'demand': demand,
#             'route': f"{origin_name} -> {dest_name}",
#             'operator': bus_operator,
#             'date': date_str,
#             'model_used': forecaster.models.get('best_name', 'default')
#         })
    
#     except Exception as e:
#         print(f"Forecast error: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# def main():
#     """Main function with enhanced training"""
#     print("🚌 Enhanced RedBus Demand Forecasting System v2.0")
#     print("=" * 60)
    
#     # Initialize forecaster
#     forecaster_instance = ImprovedRedBusDemandForecaster()
    
#     # Generate enhanced training data
#     print("Generating enhanced synthetic training data...")
#     df = forecaster_instance.generate_enhanced_synthetic_data(20000)
#     print(f"Generated {len(df)} enhanced training samples")
    
#     # Display enhanced data info
#     print("\nEnhanced Data Summary:")
#     print(df[['demand', 'distance', 'route_type', 'origin_population', 'seasonal_factor']].describe())
    
#     # Train enhanced models
#     print("\n" + "=" * 60)
#     results = forecaster_instance.train_enhanced_models(df)
    
#     # Test enhanced predictions
#     print("\n" + "=" * 60)
#     print("Testing enhanced predictions:")
    
#     test_cases = [
#         ('01', '05', 'redbus', '2024-12-25'),      # Mumbai -> Hyderabad on Christmas
#         ('03', '04', 'abhibus', '2024-11-15'),     # Bangalore -> Chennai on weekday
#         ('02', '09', 'makemytrip', '2024-10-12'),  # Delhi -> Jaipur on weekend
#         ('06', '01', 'yatra', '2024-08-15'),       # Pune -> Mumbai on Independence Day
#         ('07', '16', 'goibibo', '2024-07-20'),     # Kolkata -> Visakhapatnam
#     ]
    
#     for origin, dest, operator, date in test_cases:
#         demand = forecaster_instance.predict_demand(origin, dest, operator, date)
#         origin_name = forecaster_instance.cities.get(origin, {}).get('name', 'Unknown')
#         dest_name = forecaster_instance.cities.get(dest, {}).get('name', 'Unknown')
#         print(f"{origin_name} -> {dest_name} ({operator}) on {date}: {demand} passengers")
    
#     # Save enhanced model
#     forecaster_instance.save_model()
    
#     # Set global forecaster
#     global forecaster
#     forecaster = forecaster_instance
    
#     print("\n" + "=" * 60)
#     print("🎯 Enhanced model training completed!")
#     print("💾 Model saved successfully!")
#     print(f"📊 Best model: {forecaster.models['best_name']}")
#     print("\n🚀 Starting Flask API server...")
#     print("📡 API Endpoints:")
#     print("   GET  / - API info")
#     print("   GET  /health - Health check")
#     print("   POST /forecast - Get demand forecast")
    
#     # Start Flask app
#     app.run(debug=True, host='0.0.0.0', port=5000)

# if __name__ == "__main__":
#     main()