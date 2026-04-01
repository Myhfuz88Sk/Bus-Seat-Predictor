# import pandas as pd
# import os

# def diagnose_date_formats():
#     data_dir = 'data'
#     train_path = os.path.join(data_dir, 'train.csv')
#     test_path = os.path.join(data_dir, 'test.csv')
#     transactions_path = os.path.join(data_dir, 'transactions.csv') # Assuming Book1.csv is renamed

#     print("--- Diagnosing Date Formats ---")

#     # Diagnose train.csv
#     try:
#         df_train_raw = pd.read_csv(train_path)
#         print(f"\n{train_path} 'doj' column (first 5 rows):")
#         print(df_train_raw['doj'].head())
#         print(f"{train_path} 'doj' column dtype: {df_train_raw['doj'].dtype}")
#     except FileNotFoundError:
#         print(f"Error: {train_path} not found.")
#     except Exception as e:
#         print(f"Error reading {train_path}: {e}")

#     # Diagnose test.csv
#     try:
#         df_test_raw = pd.read_csv(test_path)
#         print(f"\n{test_path} 'doj' column (first 5 rows):")
#         print(df_test_raw['doj'].head())
#         print(f"{test_path} 'doj' column dtype: {df_test_raw['doj'].dtype}")
#     except FileNotFoundError:
#         print(f"Error: {test_path} not found.")
#     except Exception as e:
#         print(f"Error reading {test_path}: {e}")

#     # Diagnose transactions.csv
#     try:
#         df_transactions_raw = pd.read_csv(transactions_path)
#         print(f"\n{transactions_path} 'doj' column (first 5 rows):")
#         print(df_transactions_raw['doj'].head())
#         print(f"{transactions_path} 'doj' column dtype: {df_transactions_raw['doj'].dtype}")
#         print(f"\n{transactions_path} 'doi' column (first 5 rows):")
#         print(df_transactions_raw['doi'].head())
#         print(f"{transactions_path} 'doi' column dtype: {df_transactions_raw['doi'].dtype}")
#     except FileNotFoundError:
#         print(f"Error: {transactions_path} not found. Remember to rename 'Book1.csv' to 'transactions.csv' if you haven't.")
#     except Exception as e:
#         print(f"Error reading {transactions_path}: {e}")

#     print("\n--- Diagnosis Complete ---")

# if __name__ == '__main__':
#     diagnose_date_formats()




































import os
import pandas as pd
from datetime import datetime, timedelta # Import timedelta for date calculations
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration ---
DATA_DIR = 'data'
TRANSACTIONS_PATH = os.path.join(DATA_DIR, 'transactions.csv')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
MODEL_PATH = 'trained_model.pkl'
SUBMISSION_DIR = 'submission'
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission_file.csv')

# Define features used by the model - NOW INCLUDING TRANSACTION-BASED FEATURES
FEATURES = [
    'srcid', 'destid', 'day_of_week', 'day_of_month', 'month', 'year',
    'week_of_year', 'is_weekend', 'cumsum_seatcount_15', 'cumsum_searchcount_15'
]

# Global variables for API mode
trained_model = None
df_transactions_global = None

# --- Data Loading (Combined from src/data_loader.py) ---
def load_data_unified(train_file, test_file, transactions_file):
    """
    Loads and preprocesses data from CSV files.
    """
    print(f"Loading data from {train_file}, {test_file}, {transactions_file}...")
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        df_transactions = pd.read_csv(transactions_file)

        df_train['doj'] = pd.to_datetime(df_train['doj'])
        df_test['doj'] = pd.to_datetime(df_test['doj'])
        # For transactions, 'date' is the date of transaction, 'doj' is date of journey
        df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce')
        df_transactions['doj'] = pd.to_datetime(df_transactions['doj'], errors='coerce')

        df_transactions.dropna(subset=['date', 'doj'], inplace=True)

        print("Data loaded successfully.")
        return df_train, df_test, df_transactions
    except FileNotFoundError as e:
        print(f"Error: One or more data files not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading or parsing data: {e}")
        return None, None, None

# --- Feature Engineering (Combined from src/feature_engineering.py) ---
def create_time_features(df):
    """
    Creates time-based features from 'doj' column.
    """
    if 'doj' not in df.columns:
        raise ValueError("DataFrame must contain a 'doj' column for time feature engineering.")
    
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['day_of_month'] = df['doj'].dt.day
    df['month'] = df['doj'].dt.month
    df['year'] = df['doj'].dt.year
    df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = ((df['doj'].dt.dayofweek == 5) | (df['doj'].dt.dayofweek == 6)).astype(int)
    return df

def merge_transactions_data(df, df_transactions):
    """
    Merges transaction data to create cumulative features (e.g., dbd=15).
    """
    df = create_time_features(df.copy()) # Apply time features to main df first

    if df_transactions is not None and not df_transactions.empty:
        # Calculate Days Before Departure (dbd) for transactions
        # dbd = (Journey Date - Transaction Date).dt.days
        df_transactions['dbd'] = (df_transactions['doj'] - df_transactions['date']).dt.days

        # Filter transactions for bookings made at least 15 days before departure
        # (This is a common feature for early booking trends)
        df_transactions_dbd15 = df_transactions[df_transactions['dbd'] >= 15].copy()

        if not df_transactions_dbd15.empty:
            # Aggregate cumulative seatcount and searchcount for each route and DOJ
            # Use 'sourceid' and 'destinationid' from transactions as 'srcid' and 'destid'
            transactions_agg = df_transactions_dbd15.groupby(['doj', 'sourceid', 'destinationid']).agg(
                cumsum_seatcount_15=('seatcount', 'sum'),
                cumsum_searchcount_15=('searchcount', 'sum')
            ).reset_index()

            # Rename columns to match df for merging
            transactions_agg.rename(columns={
                'sourceid': 'srcid',
                'destinationid': 'destid'
            }, inplace=True)

            # Merge these features into the main dataframe (df_train or df_test)
            df = pd.merge(df, transactions_agg, on=['doj', 'srcid', 'destid'], how='left')
        else:
            print("Warning: No transactions found for dbd >= 15. Cumulative features will be all NaNs.")
    else:
        print("Warning: No transactions data provided or it is empty. Cumulative features will be all NaNs.")

    # Fill NaN values for the new cumulative features with 0 (assuming no transactions means 0 demand/searches)
    df['cumsum_seatcount_15'] = df['cumsum_seatcount_15'].fillna(0)
    df['cumsum_searchcount_15'] = df['cumsum_searchcount_15'].fillna(0)

    return df

# --- Model Definition (Combined from src/model.py) ---
def get_model():
    """
    Returns an initialized Random Forest Regressor model.
    """
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# --- Training Logic (Combined from src/train.py) ---
def train_model_unified(df_train, df_transactions, features):
    """
    Trains the demand forecasting model.
    """
    print("Preparing data for training...")
    df_train_fe = merge_transactions_data(df_train.copy(), df_transactions)
    
    # Check if all required features exist
    missing_features = [f for f in features if f not in df_train_fe.columns]
    if missing_features:
        raise ValueError(f"Missing features in training data after engineering: {missing_features}. Please check feature list and engineering steps.")
    
    X = df_train_fe[features]
    y = df_train_fe['demand']

    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
    
    y = y.fillna(y.mean())

    if X.isnull().any().any() or y.isnull().any():
        print("Warning: NaNs still present in features or target after handling. Proceeding with imputation.")

    print("Training model...")
    model = get_model()
    model.fit(X, y)
    print("Model training complete.")
    return model

# --- Prediction Logic (Combined from src/predict.py) ---
def predict_demand_unified(trained_model, df_test, df_transactions, features):
    """
    Generates predictions using the trained model.
    """
    print("Preparing test data for prediction...")
    df_test_fe = merge_transactions_data(df_test.copy(), df_transactions)

    missing_features = [f for f in features if f not in df_test_fe.columns]
    if missing_features:
        raise ValueError(f"Missing features in test data after engineering: {missing_features}. Cannot predict.")
    
    X_test = df_test_fe[features]

    for col in X_test.columns:
        if X_test[col].dtype in ['int64', 'float64']:
            X_test[col] = X_test[col].fillna(0) # Filling with 0 for prediction set
        else:
            X_test[col] = X_test[col].fillna('missing')

    if X_test.isnull().any().any():
        print("Warning: NaNs still present in test features after handling. Proceeding with imputation.")

    print("Generating predictions...")
    predictions = trained_model.predict(X_test)

    submission_df = pd.DataFrame({
        'id': df_test['id'],
        'demand': predictions.round().astype(int)
    })
    print("Predictions generated.")
    return submission_df

# --- API Resources Loading for Flask (Combined from app.py) ---
def load_api_resources():
    """
    Loads the trained model and transactions data for the Flask API.
    """
    global trained_model, df_transactions_global

    print("Loading model and transactions data for API...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            trained_model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")

        # Load transactions data for API to use in feature engineering a single point
        _, _, df_transactions_global = load_data_unified(
            TRAIN_PATH, # Dummy path
            TEST_PATH,   # Dummy path
            TRANSACTIONS_PATH
        )
        if df_transactions_global is None:
            raise Exception("Failed to load transactions data for API.")
        print(f"Transactions data loaded from {TRANSACTIONS_PATH}")

    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}. Ensure '{MODEL_PATH}' and '{TRANSACTIONS_PATH}' exist.")
        trained_model = None
        df_transactions_global = None
    except Exception as e:
        print(f"An error occurred while loading resources for API: {e}")
        trained_model = None
        df_transactions_global = None

# --- Flask API Endpoint (Combined from app.py) ---
@app.route('/predict', methods=['POST'])
def predict_api():
    """
    API endpoint to receive demand forecasting requests.
    Expects JSON input with 'doj', 'srcid', 'destid'.
    """
    if trained_model is None or df_transactions_global is None:
        return jsonify({"error": "Model or transactions data not loaded. Server is not ready."}), 500
    
    data = request.get_json(force=True)
    
    doj_str = data.get('doj')
    srcid = data.get('srcid')
    destid = data.get('destid')

    if not all([doj_str, srcid is not None, destid is not None]):
        return jsonify({"error": "Missing input parameters. Required: doj (YYYY-MM-DD), srcid (number), destid (number)."}), 400

    try:
        doj_dt = datetime.strptime(doj_str, '%Y-%m-%d')
        
        prediction_df = pd.DataFrame([{
            'doj': doj_dt,
            'srcid': int(srcid),
            'destid': int(destid)
        }])

        # Need to provide df_transactions_global for feature engineering in API context
        prediction_df = merge_transactions_data(prediction_df, df_transactions_global)
        
        # Check for missing features after engineering
        missing_features_api = [f for f in FEATURES if f not in prediction_df.columns]
        if missing_features_api:
            return jsonify({"error": f"Internal Error: Missing features for prediction after engineering: {missing_features_api}"}), 500

        X_predict = prediction_df[FEATURES]

        # Handle NaNs in the single prediction row using 0 or other appropriate fill
        for col in X_predict.columns:
            if X_predict[col].dtype in ['int64', 'float64']:
                X_predict[col] = X_predict[col].fillna(0)
            else:
                 X_predict[col] = X_predict[col].fillna('missing')

        predicted_seat_count = trained_model.predict(X_predict)[0]

        return jsonify({"predicted_seat_count": float(predicted_seat_count)}), 200

    except ValueError as ve:
        print(f"ValueError during API prediction: {ve}")
        return jsonify({"error": f"Invalid input format: {ve}. Ensure date is YYYY-MM-DD and IDs are numbers."}), 400
    except Exception as e:
        print(f"Unhandled error during API prediction: {e}")
        return jsonify({"error": "An internal server error occurred during prediction. Check server logs."}), 500

# --- Main Execution Logic ---
if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Check if the trained model exists
    if not os.path.exists(MODEL_PATH):
        print("--- Trained model not found. Running Training Pipeline ---")
        df_train, df_test, df_transactions = load_data_unified(TRAIN_PATH, TEST_PATH, TRANSACTIONS_PATH)

        if df_train is None or df_test is None or df_transactions is None:
            print("Training pipeline aborted due to data loading failure. Cannot start API.")
        else:
            trained_model_pipeline = train_model_unified(df_train, df_transactions, FEATURES)
            if trained_model_pipeline:
                try:
                    with open(MODEL_PATH, 'wb') as f:
                        pickle.dump(trained_model_pipeline, f)
                    print(f"Trained model saved to {MODEL_PATH}")
                except Exception as e:
                    print(f"Error saving model: {e}")

                submission_df = predict_demand_unified(trained_model_pipeline, df_test, df_transactions, FEATURES)
                submission_df.to_csv(SUBMISSION_FILE, index=False)
                print(f"Submission file saved to {SUBMISSION_FILE}")
            else:
                print("Model training failed. Cannot start API.")
    else:
        print("--- Trained model found. Skipping training. ---")

    # Proceed to start the Flask API server
    print("--- Starting Flask API Server ---")
    with app.app_context():
        load_api_resources()
    
    if trained_model is None or df_transactions_global is None:
        print("API cannot start because model or transactions data could not be loaded. Check previous logs.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)



