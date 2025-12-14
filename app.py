import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np

# Page Config
st.set_page_config(
    page_title="BMW Sales Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'BMW_Car_Sales_Classification.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bmw_sales_classifier.joblib')
HERO_IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'images', 'hero.png')

# Load Data
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# Custom CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #3b8ed0; /* BMW Blue-ish */
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Cards/Metrics */
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b8ed0;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2a6fa8;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("BMW Intelligence")
page = st.sidebar.radio("Navigate", ["Home", "Market Analytics", "Sales Predictor", "About"])

if page == "Home":
    st.title("BMW Sales Intelligence Platform")
    
    if os.path.exists(HERO_IMAGE_PATH):
        st.image(HERO_IMAGE_PATH, use_container_width=True)
    
    st.markdown("""
    ### Welcome to the Future of Automotive Sales Analytics
    
    This platform leverages advanced machine learning to provide actionable insights into BMW sales trends and predict sales classifications with high accuracy.
    
    **Key Features:**
    - 📊 **Market Analytics**: Deep dive into sales data across regions and models.
    - 🤖 **AI Predictor**: Real-time sales classification prediction.
    - 📈 **Strategic Insights**: Data-driven decision making support.
    """)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Average Price", f"${df['Price_USD'].mean():,.0f}")
        with col3:
            st.metric("Top Region", df['Region'].mode()[0])

elif page == "Market Analytics":
    st.title("Market Analytics 📊")
    
    if df is not None:
        # Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Volume by Region")
            fig_region = px.bar(df.groupby('Region')['Sales_Volume'].sum().reset_index(), 
                                x='Region', y='Sales_Volume', color='Region',
                                template="plotly_dark")
            st.plotly_chart(fig_region, use_container_width=True)
            
        with col2:
            st.subheader("Price Distribution by Model")
            fig_price = px.box(df, x='Model', y='Price_USD', color='Model', template="plotly_dark")
            st.plotly_chart(fig_price, use_container_width=True)
            
        # Row 2
        st.subheader("Sales Classification Distribution")
        fig_class = px.pie(df, names='Sales_Classification', title='High vs Low Sales Volume', 
                           color_discrete_sequence=px.colors.sequential.RdBu, template="plotly_dark")
        st.plotly_chart(fig_class, use_container_width=True)
        
        # Row 3
        st.subheader("Engine Size vs Price")
        fig_scatter = px.scatter(df, x='Engine_Size_L', y='Price_USD', color='Sales_Classification',
                                 size='Sales_Volume', hover_data=['Model'], template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.error("Data not found. Please ensure the dataset is in 'data/raw/'.")

elif page == "Sales Predictor":
    st.title("AI Sales Predictor 🤖")
    
    if model is not None:
        st.markdown("Enter vehicle specifications to predict sales classification (High/Low).")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_input = st.selectbox("Model", df['Model'].unique())
                year_input = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
                region_input = st.selectbox("Region", df['Region'].unique())
                color_input = st.selectbox("Color", df['Color'].unique())
                
            with col2:
                fuel_input = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
                trans_input = st.selectbox("Transmission", df['Transmission'].unique())
                engine_input = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
                mileage_input = st.number_input("Mileage (KM)", min_value=0, value=10000)
                price_input = st.number_input("Price (USD)", min_value=0, value=50000)

            submitted = st.form_submit_button("Predict Sales Class")
            
            if submitted:
                # Create dataframe for input
                input_data = pd.DataFrame({
                    'Model': [model_input],
                    'Year': [year_input],
                    'Region': [region_input],
                    'Color': [color_input],
                    'Fuel_Type': [fuel_input],
                    'Transmission': [trans_input],
                    'Engine_Size_L': [engine_input],
                    'Mileage_KM': [mileage_input],
                    'Price_USD': [price_input]
                })
                
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data).max()
                
                st.divider()
                if prediction == "High":
                    st.success(f"### Prediction: High Sales Volume 🚀")
                    st.write(f"Confidence: {probability:.2%}")
                else:
                    st.warning(f"### Prediction: Low Sales Volume 📉")
                    st.write(f"Confidence: {probability:.2%}")
                    
    else:
        st.warning("Model not found. Please run the training script first.")
        st.code("python src/train_model.py")

elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ### BMW Sales Classification Project
    
    This project demonstrates an end-to-end Data Science workflow:
    
    1.  **Data Ingestion**: Loading raw sales data.
    2.  **Preprocessing**: Handling categorical and numerical features using Scikit-Learn Pipelines.
    3.  **Modeling**: Training a Random Forest Classifier to predict sales volume categories.
    4.  **Deployment**: Interactive Streamlit application for visualization and inference.
    
    **Tech Stack:**
    -   Python
    -   Pandas & NumPy
    -   Scikit-Learn
    -   Plotly
    -   Streamlit
    
    *Created by Madhukar*
    """)
