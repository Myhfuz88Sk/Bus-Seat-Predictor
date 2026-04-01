import pandas as pd
import os
# from src.data_loader import load_data

def load_data(train_path, test_path, transactions_path):
    """
    Loads the train, test, and transactions datasets and cleans/converts date columns.
    """
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # NOTE: Using sep='\t' for transactions.csv based on inspection of the uploaded file.
        df_transactions = pd.read_csv(transactions_path, sep='\t') 

        # --- TRANSACTION DATA MAPPING FIXES ---
        
        # 1. Rename the 'transactions' column to 'seatcount' (assuming it represents bookings)
        if 'transactions' in df_transactions.columns and 'seatcount' not in df_transactions.columns:
            df_transactions.rename(columns={'transactions': 'seatcount'}, inplace=True)
            
        # 2. Add 'searchcount' as a column of 0s, since it's missing but required downstream
        if 'searchcount' not in df_transactions.columns:
            df_transactions['searchcount'] = 0
            
        # 3. Handle the MISSING 'doj' (Date of Journey) column in transactions.csv
        if 'doj' not in df_transactions.columns and 'date' in df_transactions.columns:
            # TEMPORARY FIX: Assign a dummy DOJ 30 days after the transaction 'date'.
            # This allows the pipeline to run and calculate DBD, but requires REAL DOJ data
            # for accurate forecasting.
            df_transactions['doj'] = pd.to_datetime(df_transactions['date'], errors='coerce') + pd.Timedelta(days=30)
        # --- END TRANSACTION DATA MAPPING FIXES ---
        
        
        # Convert date columns to datetime objects (must be done AFTER the dummy DOJ creation)
        df_train['doj'] = pd.to_datetime(df_train['doj'], errors='coerce')
        df_test['doj'] = pd.to_datetime(df_test['doj'], errors='coerce')
        
        if 'doj' in df_transactions.columns:
            df_transactions['doj'] = pd.to_datetime(df_transactions['doj'], errors='coerce')
        if 'date' in df_transactions.columns: 
            df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce')
        elif 'doi' in df_transactions.columns: 
             df_transactions['doi'] = pd.to_datetime(df_transactions['doi'], errors='coerce')

        # Drop rows where critical dates failed to load (became NaT)
        df_train.dropna(subset=['doj'], inplace=True)
        df_test.dropna(subset=['doj'], inplace=True)
        
        if 'doj' in df_transactions.columns:
            df_transactions.dropna(subset=['doj'], inplace=True) 

        return df_train, df_test, df_transactions
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure all data files are in the 'data/' directory.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}.")
        return None, None, None