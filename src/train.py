# import pandas as pd
# from src.feature_engineering import create_time_features, merge_transactions_data
# from src.model import get_model

# def train_model(df_train, df_transactions, features):
#     """
#     Trains the demand forecasting model.

#     Args:
#         df_train (pd.DataFrame): The training DataFrame.
#         df_transactions (pd.DataFrame): The transactions DataFrame.
#         features (list): A list of feature column names to use for training.

#     Returns:
#         sklearn.ensemble.RandomForestRegressor: The trained model.
#     """
#     df_train = create_time_features(df_train)

    
#     df_train_processed = merge_transactions_data(df_train, df_transactions, dbd_value=30)

   
#     if 'cumsum_seatcount' in features and 'cumsum_searchcount' in features:
#         df_train_processed['cumsum_seatcount'] = df_train_processed['cumsum_seatcount'].fillna(0)
#         df_train_processed['cumsum_searchcount'] = df_train_processed['cumsum_searchcount'].fillna(0)
#     elif 'cumsum_seatcount' in features: 
#         df_train_processed['cumsum_seatcount'] = df_train_processed['cumsum_seatcount'].fillna(0)
#     elif 'cumsum_searchcount' in features:
#         df_train_processed['cumsum_searchcount'] = df_train_processed['cumsum_searchcount'].fillna(0)


#     X_train = df_train_processed[features]
#     y_train = df_train_processed['final_seatcount']

#     model = get_model()
#     model.fit(X_train, y_train)
#     return model




















































# import os
# import pandas as pd
# from datetime import datetime
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from src.model import get_model

# app = Flask(__name__)
# CORS(app) 


# DATA_DIR = 'data'
# TRANSACTIONS_PATH = os.path.join(DATA_DIR, 'transactions.csv')
# TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
# TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
# MODEL_PATH = 'trained_model.pkl'
# SUBMISSION_DIR = 'submission'
# SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission_file.csv')

# FEATURES = ['srcid', 'destid', 'day_of_week', 'day_of_month', 'month', 'year', 'week_of_year', 'is_weekend']

# trained_model = None
# df_transactions_global = None

# def load_data_unified(train_file, test_file, transactions_file):
#     """
#     Loads and preprocesses data from CSV files.
#     """
#     print(f"Loading data from {train_file}, {test_file}, {transactions_file}...")
#     try:
#         df_train = pd.read_csv(train_file)
#         df_test = pd.read_csv(test_file)
#         df_transactions = pd.read_csv(transactions_file)

#         df_train['doj'] = pd.to_datetime(df_train['doj'])
#         df_test['doj'] = pd.to_datetime(df_test['doj'])
#         df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce') # Handle errors

#         df_transactions.dropna(subset=['date'], inplace=True)

#         print("Data loaded successfully.")
#         return df_train, df_test, df_transactions
#     except FileNotFoundError as e:
#         print(f"Error: One or more data files not found: {e}")
#         return None, None, None
#     except Exception as e:
#         print(f"Error loading or parsing data: {e}")
#         return None, None, None

# def create_time_features(df):
#     """
#     Creates time-based features from 'doj' column.
#     """
#     if 'doj' not in df.columns:
#         raise ValueError("DataFrame must contain a 'doj' column for time feature engineering.")
    
#     df['day_of_week'] = df['doj'].dt.dayofweek
#     df['day_of_month'] = df['doj'].dt.day
#     df['month'] = df['doj'].dt.month
#     df['year'] = df['doj'].dt.year
#     df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
#     df['is_weekend'] = ((df['doj'].dt.dayofweek == 5) | (df['doj'].dt.dayofweek == 6)).astype(int)
#     return df

# def merge_transactions_data(df, df_transactions):
#     """
#     Merges transaction data to create cumulative features.
#     (Note: This function is primarily for training/batch prediction,
#     and its current implementation only applies to the features defined in `FEATURES` global.)
#     """
#     df = create_time_features(df.copy()) 

#     return df

# def get_model():
#     """
#     Returns an initialized Random Forest Regressor model.
#     """
#     return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# def train_model(df_train, df_transactions, features):
#     """
#     Trains the demand forecasting model.
#     """
#     print("Preparing data for training...")
#     df_train_fe = merge_transactions_data(df_train.copy(), df_transactions)
    
#     for feature in features:
#         if feature not in df_train_fe.columns:
#             print(f"Warning: Feature '{feature}' not found in training data after engineering.")
    
#     X = df_train_fe[features]
#     y = df_train_fe['demand']

#     for col in X.columns:
#         if X[col].dtype in ['int64', 'float64']:
#             X[col] = X[col].fillna(X[col].mean())
#         else:
#             X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
    
#     y = y.fillna(y.mean())

#     if X.isnull().any().any() or y.isnull().any():
#         print("Warning: NaNs still present in features or target after handling.")

#     print("Training model...")
#     model = get_model()
#     model.fit(X, y)
#     print("Model training complete.")
#     return model

# # --- Prediction Logic (Combined from src/predict.py) ---
# def predict_demand_unified(trained_model, df_test, df_transactions, features):
#     """
#     Generates predictions using the trained model.
#     """
#     print("Preparing test data for prediction...")
#     df_test_fe = merge_transactions_data(df_test.copy(), df_transactions)

#     for feature in features:
#         if feature not in df_test_fe.columns:
#             print(f"Warning: Feature '{feature}' not found in test data after engineering.")
    
#     X_test = df_test_fe[features]

#     for col in X_test.columns:
#         if X_test[col].dtype in ['int64', 'float64']:
#             X_test[col] = X_test[col].fillna(X_test[col].mean() if not X_test[col].isnull().all() else 0)
#         else:
#             X_test[col] = X_test[col].fillna(X_test[col].mode()[0] if not X_test[col].mode().empty else 'missing')

#     if X_test.isnull().any().any():
#         print("Warning: NaNs still present in test features after handling.")

#     print("Generating predictions...")
#     predictions = trained_model.predict(X_test)

#     submission_df = pd.DataFrame({
#         'id': df_test['id'],
#         'demand': predictions.round().astype(int)
#     })
#     print("Predictions generated.")
#     return submission_df

# # --- API Resources Loading for Flask (Combined from app.py) ---
# def load_api_resources():
#     """
#     Loads the trained model and transactions data for the Flask API.
#     """
#     global trained_model, df_transactions_global

#     print("Loading model and transactions data for API...")
#     try:
#         with open(MODEL_PATH, 'rb') as f:
#             trained_model = pickle.load(f)
#         print(f"Model loaded from {MODEL_PATH}")

#         _, _, df_transactions_global = load_data_unified(
#             TRAIN_PATH,
#             TEST_PATH,
#             TRANSACTIONS_PATH
#         )
#         if df_transactions_global is None:
#             raise Exception("Failed to load transactions data for API.")
#         print(f"Transactions data loaded from {TRANSACTIONS_PATH}")

#     except FileNotFoundError as e:
#         print(f"Error: Required file not found - {e}. Ensure '{MODEL_PATH}' and '{TRANSACTIONS_PATH}' exist.")
#         trained_model = None
#         df_transactions_global = None
#     except Exception as e:
#         print(f"An error occurred while loading resources for API: {e}")
#         trained_model = None
#         df_transactions_global = None

# @app.route('/predict', methods=['POST'])
# def predict_api():
#     """
#     API endpoint to receive demand forecasting requests.
#     Expects JSON input with 'doj', 'srcid', 'destid'.
#     """
#     if trained_model is None:
#         return jsonify({"error": "Model not loaded. Server is not ready."}), 500
    
#     data = request.get_json(force=True)
    
#     doj_str = data.get('doj')
#     srcid = data.get('srcid')
#     destid = data.get('destid')

#     if not all([doj_str, srcid is not None, destid is not None]):
#         return jsonify({"error": "Missing input parameters. Required: doj (YYYY-MM-DD), srcid (number), destid (number)."}), 400

#     try:
#         doj_dt = datetime.strptime(doj_str, '%Y-%m-%d')
        
#         prediction_df = pd.DataFrame([{
#             'doj': doj_dt,
#             'srcid': int(srcid),
#             'destid': int(destid)
#         }])

#         prediction_df = create_time_features(prediction_df)
        
#         X_predict = prediction_df[FEATURES]

#         for col in X_predict.columns:
#             if X_predict[col].dtype in ['int64', 'float64']:
#                 X_predict[col] = X_predict[col].fillna(0)
#             else:
#                  X_predict[col] = X_predict[col].fillna('missing')

#         predicted_seat_count = trained_model.predict(X_predict)[0]

#         return jsonify({"predicted_seat_count": float(predicted_seat_count)}), 200

#     except ValueError as ve:
#         print(f"ValueError during API prediction: {ve}")
#         return jsonify({"error": f"Invalid input format: {ve}. Ensure date is Walpole-MM-DD and IDs are numbers."}), 400
#     except Exception as e:
#         print(f"Unhandled error during API prediction: {e}")
#         return jsonify({"error": "An internal server error occurred during prediction. Check server logs."}), 500

# # --- Main Execution Logic ---
# if __name__ == "__main__":
#     os.makedirs(SUBMISSION_DIR, exist_ok=True)

#     if not os.path.exists(MODEL_PATH):
#         print("--- Trained model not found. Running Training Pipeline ---")
#         df_train, df_test, df_transactions = load_data_unified(TRAIN_PATH, TEST_PATH, TRANSACTIONS_PATH)

#         if df_train is None or df_test is None or df_transactions is None:
#             print("Training pipeline aborted due to data loading failure. Cannot start API.")
#         else:
#             trained_model_pipeline = train_model(df_train, df_transactions, FEATURES)
#             if trained_model_pipeline:
#                 try:
#                     with open(MODEL_PATH, 'wb') as f:
#                         pickle.dump(trained_model_pipeline, f)
#                     print(f"Trained model saved to {MODEL_PATH}")
#                 except Exception as e:
#                     print(f"Error saving model: {e}")

#                 # Optional: Generate and save predictions for test data after training
#                 # This part is for the original pipeline, can be removed if not needed for API-only
#                 submission_df = predict_demand_unified(trained_model_pipeline, df_test, df_transactions, FEATURES)
#                 submission_df.to_csv(SUBMISSION_FILE, index=False)
#                 print(f"Submission file saved to {SUBMISSION_FILE}")
#             else:
#                 print("Model training failed. Cannot start API.")
#     else:
#         print("--- Trained model found. Skipping training. ---")

#     # Proceed to start the Flask API server
#     print("--- Starting Flask API Server ---")
#     with app.app_context():
#         load_api_resources()
    
#     if trained_model is None:
#         print("API cannot start because model or transactions data could not be loaded. Check previous logs.")
#     else:
#         app.run(debug=True, host='0.0.0.0', port=5000)




























# import os
# import pickle
# import pandas as pd
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from math import sqrt

# DATA_DIR = "data"
# MODEL_PATH = "trained_model.pkl"

# def load_data():
#     train_path = os.path.join(DATA_DIR, "train.csv")
#     test_path = os.path.join(DATA_DIR, "test.csv")
#     sub_path = os.path.join(DATA_DIR, "submission_file.csv")

#     print("📂 Loading data...")
#     train = pd.read_csv(train_path)
#     test = pd.read_csv(test_path)
#     sub = pd.read_csv(sub_path)

#     print(f"✅ Train shape: {train.shape}, Test shape: {test.shape}, Submission shape: {sub.shape}")
#     return train, test, sub

# def create_features(df):
#     """Automatically create numeric features if date columns exist"""
#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#         df['day'] = df['date'].dt.day
#         df['month'] = df['date'].dt.month
#         df['weekday'] = df['date'].dt.weekday
#     return df

# def train_model(train_df):
#     print("🚀 Starting training pipeline...")

#     # Drop non-numeric and target-unrelated columns
#     drop_cols = ['date', 'id', 'journey_id', 'route_id']
#     drop_cols = [c for c in drop_cols if c in train_df.columns]
#     target_col = 'total_seats' if 'total_seats' in train_df.columns else train_df.columns[-1]

#     train_df = create_features(train_df)
#     X = train_df.drop(drop_cols + [target_col], axis=1, errors='ignore')
#     y = train_df[target_col]

#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_valid)
#     rmse = sqrt(mean_squared_error(y_valid, preds))
#     print(f"✅ Model trained successfully | Validation RMSE: {rmse:.4f}")

#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)
#     print(f"💾 Model saved as '{MODEL_PATH}'")

#     return model

# def main():
#     if os.path.exists(MODEL_PATH):
#         print(f"✅ Model already exists at {MODEL_PATH}, skipping training.")
#         return

#     try:
#         train, test, sub = load_data()
#         train_model(train)
#         print("🎉 Training completed successfully!")
#     except Exception as e:
#         print(f"❌ Error during training: {e}")

# if __name__ == "__main__":
#     main()














import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.model import get_model
# Assuming this import works
from src.feature_engineering import create_time_features, merge_transactions_data 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define target column (Based on your data: doj,srcid,destid,final_seatcount)
TARGET_COL = 'final_seatcount' 

def train_model(df_train, df_transactions, features): 
    print("🚀 Starting training pipeline...")
    
    # 1. Generate Base Features (Time Features)
    # FIX: Explicitly call create_time_features on the copy of the training data
    df_train_fe = create_time_features(df_train.copy())
    
    # 2. Merge transaction features (dbd=15)
    # This now runs on the DataFrame which already contains the time features
    df_train_fe = merge_transactions_data(df_train_fe, df_transactions, dbd_value=15)
    
    # Check if the target column exists (it should now, as it's not modified by FE)
    if TARGET_COL not in df_train_fe.columns:
        # This safeguard is useful if the target column name changes
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data. Check data or define TARGET_COL.")

    # 3. Drop non-feature columns
    # Ensure raw date column 'doj' is dropped to prevent string-to-float error
    drop_cols = ['doj', 'id', 'journey_id', 'route_key'] 
    drop_cols = [c for c in drop_cols if c in df_train_fe.columns and c != TARGET_COL]

    # 4. Prepare X and y
    # This line now succeeds because the features are present in df_train_fe
    X = df_train_fe[features] 
    y = df_train_fe[TARGET_COL]

    # Drop any remaining non-numeric columns from features
    X = X.select_dtypes(include=['number'])

    # Handle NaNs (Imputation is crucial)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # 5. Training
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, preds))
    print(f"✅ Model trained successfully | Validation RMSE: {rmse:.4f}")

    return model

# Note: The model saving logic is handled by run_pipeline.py