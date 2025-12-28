import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(df, n_clusters=3):
    """
    Performs K-Means clustering on the customer data.
    Uses 'Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases'.
    """
    # Select features for clustering
    features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    
    # Drop rows with missing values in these columns just in case
    df_clean = df.dropna(subset=features).copy()
    
    if df_clean.empty:
        return df
    
    X = df_clean[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df_clean
