import os
import hashlib
import pickle
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from database import get_connection, create_users_table
from src.data_loader import load_data  # Make sure this exists and works

# -------------------------------
# App Setup
# -------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Replace with env var in production

# Ensure users table exists
create_users_table()

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'enhanced_redbus_model.pkl')

# -------------------------------
# Load Model
# -------------------------------
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# -------------------------------
# Helper Functions
# -------------------------------
def engineer_features(df):
    """
    Transform input dataframe into the features your model expects.
    Replace this with your actual preprocessing from training.
    """
    df = df.copy()
    df['doj'] = pd.to_datetime(df['doj'], errors='coerce')
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    return df[['src', 'dest', 'day_of_week', 'month']]

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/district.html')
def district_page():
    return render_template('district.html')

@app.route('/state.html')
def state_page():
    return render_template('state.html')

@app.route('/analasys.html')
def analasys_page():
    return render_template('analasys.html')

@app.route('/main.html')
def main_page():
    return render_template('main.html')

# -------------------------------
# Prediction API
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Prediction not available."})

    data = request.get_json()
    doj = data.get('doj')
    src = data.get('src')
    dest = data.get('dest')

    # Load CSV datasets
    train_path = os.path.join(BASE_DIR, 'data', 'train.csv')
    test_path = os.path.join(BASE_DIR, 'data', 'test.csv')
    transactions_path = os.path.join(BASE_DIR, 'data', 'transactions.csv')

    train_df, test_df, transactions_df = load_data(train_path, test_path, transactions_path)

    if train_df is None or test_df is None or transactions_df is None:
        return jsonify({"error": "Failed to load dataset files."})

    # Create single-row DataFrame for prediction
    input_df = pd.DataFrame([{'doj': doj, 'src': src, 'dest': dest}])
    features_df = engineer_features(input_df)

    try:
        predicted_demand = model.predict(features_df)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"})

    prediction_result = {
        "demand": int(predicted_demand),
        "peakTime": "6PM",
        "occupancy": f"{min(int(predicted_demand / 50 * 100), 100)}%",
        "buses": max(int(predicted_demand / 50), 1),
        "priceRange": "₹800-1200"
    }
    return jsonify(prediction_result)

# -------------------------------
# Signup
# -------------------------------
@app.route('/signup_submit', methods=['POST'])
def signup_submit():
    try:
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        usertype = request.form.get('usertype')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('signup_page'))

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already registered.", "error")
            conn.close()
            return redirect(url_for('signup_page'))

        cursor.execute(
            "INSERT INTO users (full_name, email, user_type, password) VALUES (%s, %s, %s, %s)",
            (fullname, email, usertype, hashed_password)
        )
        conn.commit()
        conn.close()

        flash("Account created successfully!", "success")
        return redirect(url_for('login_page'))

    except Exception as e:
        print("Signup error:", e)
        flash("Internal server error.", "error")
        return redirect(url_for('signup_page'))

# -------------------------------
# Login
# -------------------------------
@app.route('/login_submit', methods=['POST'])
def login_submit():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email=%s AND password=%s",
            (email, hashed_password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            flash(f"🎉 Welcome, {user['full_name']}!", "success")
            return redirect(url_for('state_page'))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for('login_page'))

    except Exception as e:
        print("Login error:", e)
        flash("Internal server error. Please try again.", "error")
        return redirect(url_for('login_page'))

# -------------------------------
# Forecast
# -------------------------------
@app.route('/forecast', methods=['POST'])
def forecast():
    city = request.form.get('city')
    api_key = os.environ.get("WEATHER_API_KEY")  # Use env var for safety
    if not api_key:
        return jsonify({"error": "Weather API key not set."})

    try:
        api_url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=1"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        forecast_info = {
            "location": data['location']['name'],
            "region": data['location']['region'],
            "country": data['location']['country'],
            "temp_c": data['current']['temp_c'],
            "condition": data['current']['condition']['text']
        }
        return jsonify(forecast_info)
    except requests.exceptions.RequestException as e:
        print("Forecast API error:", e)
        return jsonify({"error": "Network error. Please check your connection."})

# -------------------------------
# Main
# -------------------------------
# No debug=True here for production
if __name__ == '__main__':
    app.run()
