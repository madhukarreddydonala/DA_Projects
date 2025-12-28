from flask import Flask, jsonify, request
import sqlite3
import pandas as pd
import os
from ml_utils import perform_clustering

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'customer_revenue.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return jsonify({"message": "Customer Revenue Analytics API is running!"})

@app.route('/data')
def get_data():
    conn = get_db_connection()
    # Limit to 1000 rows for performance if needed, but dataset is small (3900 rows)
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return jsonify(df.to_dict(orient='records'))

@app.route('/metrics')
def get_metrics():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    total_revenue = df['Purchase Amount (USD)'].sum()
    avg_purchase = df['Purchase Amount (USD)'].mean()
    total_customers = df['Customer ID'].nunique()
    
    metrics = {
        "total_revenue": float(total_revenue),
        "avg_purchase_value": float(avg_purchase),
        "total_customers": int(total_customers)
    }
    return jsonify(metrics)

@app.route('/category_sales')
def get_category_sales():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT Category, SUM(\"Purchase Amount (USD)\") as Revenue FROM transactions GROUP BY Category", conn)
    conn.close()
    return jsonify(df.to_dict(orient='records'))

@app.route('/segmentation')
def get_segmentation():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    # Perform clustering
    clustered_df = perform_clustering(df)
    
    return jsonify(clustered_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
