import pandas as pd
from src.feature_engineering import create_time_features, merge_transactions_data

def predict_demand(model, df_test, df_transactions, features): 
    """
    Generates demand predictions on the test dataset.
    """
    # 1. Feature Engineering on Test Data
    # Apply time features
    df_test_processed = create_time_features(df_test.copy()) 

    # Merge transactions (dbd=15 must match training)
    # The merge_transactions_data function should handle calling create_time_features internally.
    df_test_processed = merge_transactions_data(df_test_processed, df_transactions, dbd_value=15)

    # 2. Prepare Features
    # Ensure features exist, fill NaNs (with 0 is common for test/prediction time)
    X_test = df_test_processed[features]
    X_test = X_test.select_dtypes(include=['number']).fillna(0) 

    # 3. Prediction
    predictions = model.predict(X_test)

    # 4. Create Submission DataFrame
    # FIX: Use 'route_key' as the unique identifier for the submission file.
    submission_df = pd.DataFrame({
        'route_key': df_test['route_key'], # <-- CHANGED from 'id' to 'route_key'
        'final_seatcount': predictions.round().astype(int)
    })
    return submission_df