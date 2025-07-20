import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os

def create_classification_model():
    """Create and train an Iris classification model"""
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Classification Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Save the model
    model_data = {
        'model': rf_classifier,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'model_type': 'classification',
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, 'models/iris_classifier.pkl')
    return model_data

def create_regression_model():
    """Create and train a house price prediction model"""
    # Create synthetic housing data similar to California housing
    print("Creating synthetic housing dataset...")
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic features
    med_inc = np.random.uniform(0.5, 15.0, n_samples)
    house_age = np.random.uniform(1.0, 52.0, n_samples)
    ave_rooms = np.random.uniform(1.0, 20.0, n_samples)
    ave_bedrms = np.random.uniform(0.5, 5.0, n_samples)
    population = np.random.uniform(3.0, 35000.0, n_samples)
    ave_occup = np.random.uniform(0.5, 20.0, n_samples)
    latitude = np.random.uniform(32.0, 42.0, n_samples)
    longitude = np.random.uniform(-125.0, -114.0, n_samples)
    
    # Stack features
    X = np.column_stack([med_inc, house_age, ave_rooms, ave_bedrms, 
                        population, ave_occup, latitude, longitude])
    
    # Create target with realistic relationships
    y = (med_inc * 0.8 +                    # Income is strong predictor
         ave_rooms * 0.3 +                  # More rooms = higher price
         -house_age * 0.05 +                # Older houses cheaper
         -ave_bedrms * 0.2 +                # Fewer bedrooms per room = better
         latitude * 0.1 +                   # Northern areas might be pricier
         -longitude * 0.05 +                # Eastern areas might be pricier
         np.random.normal(0, 0.5, n_samples))  # Add noise
    
    # Ensure positive prices
    y = np.abs(y) + 0.5
    
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                    'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Model R² Score: {r2:.4f}")
    print(f"Regression Model MSE: {mse:.4f}")
    
    # Save the model
    model_data = {
        'model': rf_regressor,
        'feature_names': feature_names,
        'model_type': 'regression',
        'r2_score': r2,
        'mse': mse
    }
    
    joblib.dump(model_data, 'models/housing_regressor.pkl')
    return model_data

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Training Classification Model...")
    classification_model = create_classification_model()
    
    print("\nTraining Regression Model...")
    regression_model = create_regression_model()
    
    print("\nModels saved successfully!")
