import pandas as pd
import numpy as np 

def create_time_features(df):
    """
    Creates time-based features from the 'doj' column.
    """
    if 'doj' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['doj']):
        # If doj is missing or wrong type, this function cannot proceed
        return df 
        
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['day_of_month'] = df['doj'].dt.day
    df['month'] = df['doj'].dt.month
    df['year'] = df['doj'].dt.year
    df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['doj'].dt.dayofweek >= 5).astype(int) 
    return df

def merge_transactions_data(df_main, df_transactions, dbd_value=15):
    """
    Calculates cumulative transaction features (seatcount and searchcount)
    for bookings made up to 'dbd_value' days before departure and merges them 
    with the main dataframe (train/test).
    """
    df_transactions = df_transactions.copy()
    
    # 1. Calculate Days Before Departure (dbd)
    
    # FIX: Explicitly check for 'date' first, as confirmed by your data inspection
    transaction_date_col = None
    if 'date' in df_transactions.columns:
        transaction_date_col = 'date'
    elif 'doi' in df_transactions.columns:
        transaction_date_col = 'doi'
    
    if not transaction_date_col or 'doj' not in df_transactions.columns:
         # This error should be avoided by the fixes in load_data.py, 
         # but remains here as a safeguard.
         raise ValueError("Transaction DataFrame must contain 'doj' and a transaction date column ('date' or 'doi').")
         
    df_transactions['dbd'] = (df_transactions['doj'] - df_transactions[transaction_date_col]).dt.days

    # 2. Filter transactions based on dbd threshold
    df_transactions_filtered = df_transactions[df_transactions['dbd'] >= dbd_value].copy()
    
    # 3. Aggregate cumulative features
    transactions_agg = df_transactions_filtered.groupby(['doj', 'srcid', 'destid']).agg(
        **{
            # 'seatcount' and 'searchcount' are now available due to renaming/injection in load_data.py
            f'cumsum_seatcount_{dbd_value}': ('seatcount', 'sum'),
            f'cumsum_searchcount_{dbd_value}': ('searchcount', 'sum')
        }
    ).reset_index()

    # 4. Merge the new features into the main dataframe
    df_merged = pd.merge(
        df_main,
        transactions_agg,
        on=['doj', 'srcid', 'destid'],
        how='left'
    )
    
    # 5. Fill NaNs 
    df_merged[f'cumsum_seatcount_{dbd_value}'] = df_merged[f'cumsum_seatcount_{dbd_value}'].fillna(0)
    df_merged[f'cumsum_searchcount_{dbd_value}'] = df_merged[f'cumsum_searchcount_{dbd_value}'].fillna(0)
    
    return df_merged