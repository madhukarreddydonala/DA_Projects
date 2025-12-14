import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'BMW_Car_Sales_Classification.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    return df

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    # Features and Target
    X = df.drop(['Sales_Classification', 'Sales_Volume'], axis=1) # Dropping Sales_Volume as it's directly correlated to classification usually, or we want to predict class based on car specs
    y = df['Sales_Classification']
    
    # Identify categorical and numerical columns
    categorical_features = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
    numerical_features = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']
    
    # Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Model Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training model...")
    clf.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(MODELS_DIR, 'bmw_sales_classifier.joblib')
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
