import os
import pickle 
from src.data_loader import load_data
from src.train import train_model
from src.predict import predict_demand

def run_pipeline():
    """
    Orchestrates the entire demand forecasting pipeline.
    """
    # Define file paths
    data_dir = 'data'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    transactions_path = os.path.join(data_dir, 'transactions.csv')

    submission_dir = 'submission'
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, 'submission_file.csv')

    print("Step 1: Loading data...")
    df_train, df_test, df_transactions = load_data(train_path, test_path, transactions_path)

    if df_train is None or df_test is None or df_transactions is None:
        print("Data loading failed. Exiting pipeline.")
        return

    # FIX: Updated features list to include the newly engineered cumulative columns
    features = [
        'srcid', 'destid', 
        'day_of_week', 'day_of_month', 'month', 'year', 'week_of_year', 'is_weekend',
        'cumsum_seatcount_15',      # <-- Feature engineered from transactions
        'cumsum_searchcount_15'     # <-- Feature engineered from transactions (as zeros)
    ]


    print("Step 2: Training model...")
    # NOTE: Assuming train_model's signature is now train_model(df_train, df_transactions, features)
    trained_model = train_model(df_train, df_transactions, features) 
    print("Model training complete.")

    model_save_path = 'trained_model.pkl' 
    try:
        with open(model_save_path, 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"Trained model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("Step 3: Generating predictions...")
    # NOTE: Assuming predict_demand's signature is now predict_demand(model, df_test, df_transactions, features)
    submission_df = predict_demand(trained_model, df_test, df_transactions, features)
    print("Predictions generated.")

    print(f"Step 4: Saving submission file to {submission_path}...")
    submission_df.to_csv(submission_path, index=False)
    print("Pipeline finished successfully! 🎉")

if __name__ == "__main__":
    run_pipeline()