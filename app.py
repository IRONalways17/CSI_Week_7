import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.datasets import load_iris

# Configure page
st.set_page_config(
    page_title="ML Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model(model_path):
    """Load the trained model"""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def load_sample_data(model_type):
    """Load sample data for demonstration"""
    if model_type == 'classification':
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    else:
        # Create sample California housing data
        np.random.seed(42)
        n_samples = 100
        data = {
            'MedInc': np.random.uniform(0.5, 15.0, n_samples),
            'HouseAge': np.random.uniform(1.0, 52.0, n_samples),
            'AveRooms': np.random.uniform(1.0, 20.0, n_samples),
            'AveBedrms': np.random.uniform(0.5, 10.0, n_samples),
            'Population': np.random.uniform(3.0, 35682.0, n_samples),
            'AveOccup': np.random.uniform(0.5, 20.0, n_samples),
            'Latitude': np.random.uniform(32.0, 42.0, n_samples),
            'Longitude': np.random.uniform(-125.0, -114.0, n_samples)
        }
        return pd.DataFrame(data)

def create_feature_importance_plot(model_data):
    """Create feature importance visualization"""
    if hasattr(model_data['model'], 'feature_importances_'):
        importances = model_data['model'].feature_importances_
        feature_names = model_data['feature_names']
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def create_prediction_confidence_plot(probabilities, target_names):
    """Create prediction confidence visualization"""
    if probabilities is not None:
        confidence_df = pd.DataFrame({
            'Class': target_names,
            'Probability': probabilities[0]
        })
        
        fig = px.bar(
            confidence_df,
            x='Class',
            y='Probability',
            title='Prediction Confidence',
            color='Probability',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Machine Learning Model Deployment</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    # Check if models exist
    classification_model_path = "models/iris_classifier.pkl"
    regression_model_path = "models/housing_regressor.pkl"
    
    available_models = []
    if os.path.exists(classification_model_path):
        available_models.append("Iris Classification")
    if os.path.exists(regression_model_path):
        available_models.append("Housing Price Prediction")
    
    if not available_models:
        st.error("⚠️ No trained models found!")
        st.info("Training models automatically... This may take a moment.")
        
        # Auto-train models for deployment
        with st.spinner("Training machine learning models..."):
            try:
                import subprocess
                import sys
                
                # Run the training script
                result = subprocess.run([sys.executable, "train_models.py"], 
                                      capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    st.success("✅ Models trained successfully! Please refresh the page.")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.stderr}")
            except Exception as e:
                st.error(f"Failed to train models: {str(e)}")
                st.info("Please run `python train_models.py` manually first.")
        return
    
    selected_model = st.sidebar.selectbox("Choose a model:", available_models)
    
    # Load the selected model
    if selected_model == "Iris Classification":
        model_data = load_model(classification_model_path)
        model_type = "classification"
    else:
        model_data = load_model(regression_model_path)
        model_type = "regression"
    
    if model_data is None:
        st.error(f"Failed to load {selected_model} model!")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Analysis", "📈 Data Exploration", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Make Predictions")
        
        if model_type == "classification":
            st.subheader("Iris Species Classification")
            st.write("Enter the flower measurements to predict the iris species:")
            
            col1, col2 = st.columns(2)
            with col1:
                sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
                petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            
            with col2:
                sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            
            # Make prediction
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            if st.button("🔮 Predict Species", type="primary"):
                prediction = model_data['model'].predict(input_features)[0]
                probabilities = model_data['model'].predict_proba(input_features)
                predicted_species = model_data['target_names'][prediction]
                confidence = np.max(probabilities) * 100
                
                # Display prediction
                st.markdown(f'<div class="prediction-box">Predicted Species: {predicted_species.title()}<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                
                # Show confidence plot
                fig = create_prediction_confidence_plot(probabilities, model_data['target_names'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regression
            st.subheader("Housing Price Prediction")
            st.write("Enter the house characteristics to predict the price:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                med_inc = st.number_input("Median Income (10k units)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                house_age = st.number_input("House Age (years)", min_value=1.0, max_value=52.0, value=15.0, step=1.0)
                ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=6.0, step=0.1)
            
            with col2:
                ave_bedrooms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
                population = st.number_input("Population", min_value=3.0, max_value=35682.0, value=3000.0, step=100.0)
                ave_occupancy = st.number_input("Average Occupancy", min_value=0.5, max_value=20.0, value=3.0, step=0.1)
            
            with col3:
                latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01)
                longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01)
            
            # Make prediction
            input_features = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms, population, ave_occupancy, latitude, longitude]])
            
            if st.button("🔮 Predict Price", type="primary"):
                prediction = model_data['model'].predict(input_features)[0]
                
                # Display prediction
                st.markdown(f'<div class="prediction-box">Predicted Price: ${prediction:.2f} (hundreds of thousands)</div>', unsafe_allow_html=True)
                
                # Show input feature values
                st.subheader("Input Summary")
                feature_df = pd.DataFrame({
                    'Feature': model_data['feature_names'],
                    'Value': input_features[0]
                })
                st.dataframe(feature_df, use_container_width=True)
    
    with tab2:
        st.header("Model Analysis")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            if model_type == "classification":
                st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
            else:
                st.metric("R² Score", f"{model_data['r2_score']:.4f}")
                st.metric("MSE", f"{model_data['mse']:.4f}")
        
        with col2:
            st.subheader("Model Details")
            st.write(f"**Model Type:** {model_data['model_type'].title()}")
            st.write(f"**Algorithm:** {type(model_data['model']).__name__}")
            st.write(f"**Features:** {len(model_data['feature_names'])}")
        
        # Feature importance plot
        fig = create_feature_importance_plot(model_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Exploration")
        
        # Load and display sample data
        sample_data = load_sample_data(model_type)
        
        st.subheader("Sample Data")
        st.dataframe(sample_data.head(10), use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        st.dataframe(sample_data.describe(), use_container_width=True)
        
        # Correlation heatmap
        if model_type == "classification":
            numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
            corr_data = sample_data[numeric_cols]
        else:
            corr_data = sample_data
        
        if len(corr_data.columns) > 1:
            st.subheader("Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        # Distribution plots
        if model_type == "classification":
            st.subheader("Feature Distributions by Species")
            feature_to_plot = st.selectbox("Select feature:", model_data['feature_names'])
            
            fig = px.box(sample_data, x='species', y=feature_to_plot, 
                        title=f'{feature_to_plot} Distribution by Species')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Model Information")
        
        if model_type == "classification":
            st.subheader("Iris Classification Model")
            st.write("""
            This model predicts the species of iris flowers based on four measurements:
            - **Sepal Length**: Length of the sepal in centimeters
            - **Sepal Width**: Width of the sepal in centimeters  
            - **Petal Length**: Length of the petal in centimeters
            - **Petal Width**: Width of the petal in centimeters
            
            The model can classify flowers into three species:
            - **Setosa**: Typically has shorter petals
            - **Versicolor**: Medium-sized petals and sepals
            - **Virginica**: Generally the largest flowers
            """)
        else:
            st.subheader("Housing Price Prediction Model")
            st.write("""
            This model predicts housing prices based on various characteristics from California housing data:
            - **MedInc**: Median income in block group (in tens of thousands of dollars)
            - **HouseAge**: Median house age in block group
            - **AveRooms**: Average number of rooms per household
            - **AveBedrms**: Average number of bedrooms per household
            - **Population**: Block group population
            - **AveOccup**: Average number of household members
            - **Latitude**: Block group latitude
            - **Longitude**: Block group longitude
            
            The model predicts the median house value in hundreds of thousands of dollars.
            """)
        
        st.subheader("How to Use")
        st.write("""
        1. **Select a Model**: Choose between classification or regression in the sidebar
        2. **Make Predictions**: Enter feature values in the Prediction tab
        3. **Analyze Results**: View model performance and feature importance
        4. **Explore Data**: Examine the underlying data patterns
        """)
        
        st.subheader("Technical Details")
        st.write(f"""
        - **Algorithm**: Random Forest
        - **Framework**: Scikit-learn
        - **Model File**: {classification_model_path if model_type == 'classification' else regression_model_path}
        - **Features**: {len(model_data['feature_names'])}
        """)

if __name__ == "__main__":
    main()
