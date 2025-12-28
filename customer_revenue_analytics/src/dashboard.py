import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page config
st.set_page_config(page_title="Customer Revenue Analytics", layout="wide")

# API URL
API_URL = "http://127.0.0.1:5000"

@st.cache_data
def fetch_data(endpoint):
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            st.error(f"Failed to fetch data from {endpoint}")
            return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask API. Make sure `src/api.py` is running.")
        return pd.DataFrame()

@st.cache_data
def fetch_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except:
        return {}

@st.cache_data
def fetch_segmentation_data():
    try:
        response = requests.get(f"{API_URL}/segmentation")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

# Title
st.title("Customer Revenue Analytics Dashboard")

# Check if API is running by fetching metrics
metrics = fetch_metrics()

if metrics:
    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${metrics.get('total_revenue', 0):,.2f}")
    col2.metric("Avg Purchase Value", f"${metrics.get('avg_purchase_value', 0):,.2f}")
    col3.metric("Total Customers", metrics.get('total_customers', 0))
else:
    st.warning("⚠️ Flask API is not running. Please run `python src/api.py` in a separate terminal.")

    st.warning("⚠️ Flask API is not running. Please run `python src/api.py` in a separate terminal.")

# Create tabs
tab1, tab2 = st.tabs(["📈 Overview", "🧠 Advanced Analytics"])

with tab1:
    # Fetch main data
    df = fetch_data("data")

    if not df.empty:
        # Sidebar filters
        st.sidebar.header("Filters")
        selected_category = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
        
        filtered_df = df[df['Category'].isin(selected_category)]

        # --- Visualizations ---
        
        # 1. Revenue by Category (Bar Chart)
        st.subheader("Revenue by Category")
        category_sales = filtered_df.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=category_sales, x='Category', y='Purchase Amount (USD)', ax=ax1, palette='viridis')
        ax1.set_title("Total Revenue per Category")
        st.pyplot(fig1)

        # 2. Distribution of Purchase Amount (Histogram)
        st.subheader("Purchase Amount Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Purchase Amount (USD)'], bins=20, kde=True, ax=ax2, color='blue')
        ax2.set_title("Distribution of Purchase Amounts")
        st.pyplot(fig2)

        # 3. Payment Method Usage (Pie Chart)
        st.subheader("Payment Method Distribution")
        payment_counts = filtered_df['Payment Method'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        ax3.set_title("Payment Methods")
        st.pyplot(fig3)

        # 4. Review Rating vs Purchase Amount (Scatter)
        st.subheader("Review Rating vs Purchase Amount")
        fig4 = px.scatter(filtered_df, x='Review Rating', y='Purchase Amount (USD)', color='Category', title="Review Rating vs Purchase Amount")
        st.plotly_chart(fig4)

        # 5. Data Table
        st.subheader("Raw Data")
        st.dataframe(filtered_df)

with tab2:
    st.header("Customer Segmentation & Advanced Analytics")
    
    seg_df = fetch_segmentation_data()
    
    if not seg_df.empty:
        st.subheader("Customer Segments (K-Means Clustering)")
        st.markdown("Clustering based on **Age**, **Purchase Amount**, **Review Rating**, and **Previous Purchases**.")
        
        # Scatter plot for clusters
        fig_cluster = px.scatter(
            seg_df, 
            x='Age', 
            y='Purchase Amount (USD)', 
            color='Cluster',
            size='Previous Purchases',
            hover_data=['Category', 'Review Rating'],
            title="Customer Segments: Age vs Purchase Amount"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Boxplot of Purchase Amount by Cluster
        st.subheader("Spending Habits by Cluster")
        fig_box, ax_box = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=seg_df, x='Cluster', y='Purchase Amount (USD)', ax=ax_box, palette='Set2')
        st.pyplot(fig_box)
        
        # Correlation Heatmap
        st.subheader("Feature Correlations")
        numeric_df = seg_df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        
    else:
        st.info("Could not load segmentation data. Ensure the API is running.")

# else:
#     st.info("No data available. Ensure the API is running and serving data.")
