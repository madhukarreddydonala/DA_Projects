import pandas as pd
import sqlite3
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'customer_shopping_behavior.csv')
DB_PATH = os.path.join(BASE_DIR, 'data', 'customer_revenue.db')

def load_data():
    """Loads data from CSV to SQLite database."""
    print(f"Loading data from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Basic cleaning (if needed)
    # df.dropna(inplace=True) 
    
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # Save to database
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Data saved to {DB_PATH}")

if __name__ == "__main__":
    load_data()
