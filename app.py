from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import hashlib
import requests
import pickle
import pandas as pd
from database import get_connection, create_users_table
from src.data_loader import load_data  # THIS IS THE CORRECT FUNCTION


app = Flask(__name__)
app.secret_key = "supersecretkey"

# Ensure users table exists
create_users_table()

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')  # Landing page

@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/district.html')
def district_page():
    return render_template('district.html')

# @app.route('/state.html')
# def state_page():
#     return render_template('state.html')

@app.route('/analasys.html')
def analasys_page():
    return render_template('analasys.html')

@app.route('/main.html')
def main_page():
    return render_template('main.html')


try:
    with open('enhanced_redbus_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)
    model = None

def engineer_features(df):
    """
    Transform input dataframe into the features your model expects.
    Replace this with your actual preprocessing from training.
    """
    df = df.copy()
    df['doj'] = pd.to_datetime(df['doj'], errors='coerce')
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    # Replace with the actual features your model uses
    return df[['src', 'dest', 'day_of_week', 'month']]


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Prediction not available."})

    data = request.get_json()
    doj = data['doj']
    src = data['src']
    dest = data['dest']

    # Optional: load datasets if needed for feature reference
    train_df, test_df, transactions_df = load_data(
        'data/train.csv',
        'data/test.csv',
        'data/transactions.csv'
    )

    if train_df is None or test_df is None or transactions_df is None:
        return jsonify({"error": "Failed to load dataset files."})

    # Create single-row DataFrame from user input
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
# Signup Submission
# -------------------------------
@app.route('/signup_submit', methods=['POST'])
def signup_submit():
    try:
        fullname = request.form['fullname']
        email = request.form['email']
        usertype = request.form['usertype']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('signup_page'))

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already registered.", "error")
            conn.close()
            return redirect(url_for('signup_page'))

        # Insert user
        cursor.execute(
            "INSERT INTO users (full_name, email, user_type, password) VALUES (%s, %s, %s, %s)",
            (fullname, email, usertype, hashed_password)
        )
        conn.commit()
        conn.close()

        flash("Account created successfully!", "success")
        return redirect(url_for('login_page'))

    except Exception as e:
        import traceback
        print("Signup route error:", e)
        traceback.print_exc()
        flash("Internal server error", "error")
        return redirect(url_for('signup_page'))

# -------------------------------
# Login Submission
# -------------------------------
@app.route('/login_submit', methods=['POST'])
def login_submit():
    try:
        email = request.form['email']
        password = request.form['password']

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
        import traceback
        print("Login route error:", e)
        traceback.print_exc()
        flash("Internal server error. Please try again.", "error")
        return redirect(url_for('login_page'))

# -------------------------------
# Main Page (After Login)
# -------------------------------
# @app.route('/main.html')
# def main_page():
#     return render_template('main.html')  # Create main.html

@app.route('/state.html')
def state_page():
    return render_template('state.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    city = request.form.get('city')
    try:
        # Replace YOUR_API_KEY with your actual weather API key
        api_url = f"http://api.weatherapi.com/v1/forecast.json?key=YOUR_API_KEY&q={city}&days=1"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        # Extract relevant info for frontend
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
        return jsonify({"error": "Network error. Please check your connection and try again."})
# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)

