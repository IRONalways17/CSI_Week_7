# Advanced ML Model Deployment Examples

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.datasets import make_classification
from plotly.subplots import make_subplots
import joblib

def create_advanced_classification_example():
    """
    Advanced classification example with more features
    """
    st.header("🧠 Advanced Classification: Customer Churn Prediction")
    
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Generate features
    data = {
        'tenure': np.random.randint(1, 72, n_customers),
        'monthly_charges': np.random.uniform(20, 120, n_customers),
        'total_charges': np.random.uniform(20, 8000, n_customers),
        'contract_length': np.random.choice([1, 12, 24], n_customers),
        'payment_method': np.random.choice([0, 1, 2, 3], n_customers),  # 0-3 for different methods
        'internet_service': np.random.choice([0, 1, 2], n_customers),  # 0: No, 1: DSL, 2: Fiber
        'tech_support': np.random.choice([0, 1], n_customers),
        'online_security': np.random.choice([0, 1], n_customers),
        'streaming_tv': np.random.choice([0, 1], n_customers),
        'paperless_billing': np.random.choice([0, 1], n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (churn) based on features
    churn_probability = (
        0.8 / (1 + np.exp(-(df['tenure'] - 24) / 12)) +  # Higher churn for short tenure
        0.3 * (df['monthly_charges'] > 80) +              # Higher churn for expensive plans
        0.2 * (df['contract_length'] == 1) +              # Higher churn for month-to-month
        0.15 * (df['tech_support'] == 0) +                # Higher churn without tech support
        np.random.normal(0, 0.1, n_customers)             # Add some noise
    )
    
    df['churn'] = (churn_probability > 0.5).astype(int)
    
    # Display dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Customers**: {len(df):,}")
        st.write(f"**Churned Customers**: {df['churn'].sum():,}")
        st.write(f"**Churn Rate**: {df['churn'].mean():.2%}")
        
    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head(), use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Analysis")
    
    # Churn distribution by categorical features
    categorical_features = ['contract_length', 'payment_method', 'internet_service', 'tech_support']
    
    fig_subplots = make_subplots(rows=2, cols=2, 
                                subplot_titles=categorical_features)
    
    for i, feature in enumerate(categorical_features):
        row = i // 2 + 1
        col = i % 2 + 1
        
        churn_by_feature = df.groupby(feature)['churn'].mean()
        
        fig_subplots.add_trace(
            go.Bar(x=churn_by_feature.index, y=churn_by_feature.values, 
                   name=feature, showlegend=False),
            row=row, col=col
        )
    
    fig_subplots.update_layout(height=400, title_text="Churn Rate by Feature")
    st.plotly_chart(fig_subplots, use_container_width=True)
    
    return df

def create_model_comparison():
    """
    Compare multiple models and show performance metrics
    """
    st.header("📊 Model Comparison Dashboard")
    
    # Generate synthetic data for comparison
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                              n_redundant=2, random_state=42)
    
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred
        }
    
    # Display model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Comparison")
        
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[name]['accuracy'] for name in results.keys()],
            'ROC AUC': [results[name]['roc_auc'] for name in results.keys()]
        })
        
        st.dataframe(performance_df, use_container_width=True)
        
        # Bar chart of performance
        fig = px.bar(performance_df, x='Model', y=['Accuracy', 'ROC AUC'], 
                    title='Model Performance Metrics', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curves")
        
        fig = go.Figure()
        
        for name, result in results.items():
            fig.add_trace(go.Scatter(
                x=result['fpr'], y=result['tpr'],
                mode='lines',
                name=f'{name} (AUC = {result["roc_auc"]:.3f})'
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_batch_prediction_interface():
    """
    Create interface for batch predictions from uploaded files
    """
    st.header("📁 Batch Prediction Interface")
    
    st.write("Upload a CSV file to make predictions on multiple samples at once.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check if the required columns exist (for iris model)
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            
            if all(col in df.columns for col in required_columns):
                # Load iris model if available
                try:
                    model_data = joblib.load('models/iris_classifier.pkl')
                    
                    # Make predictions
                    X = df[required_columns].values
                    predictions = model_data['model'].predict(X)
                    probabilities = model_data['model'].predict_proba(X)
                    
                    # Add predictions to dataframe
                    df['predicted_species'] = [model_data['target_names'][pred] for pred in predictions]
                    df['confidence'] = np.max(probabilities, axis=1)
                    
                    st.subheader("Predictions")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download predictions
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show prediction summary
                    st.subheader("Prediction Summary")
                    summary = df['predicted_species'].value_counts()
                    
                    fig = px.pie(values=summary.values, names=summary.index,
                               title="Predicted Species Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except FileNotFoundError:
                    st.error("Iris classification model not found. Please train the model first.")
            else:
                st.warning(f"CSV file must contain columns: {', '.join(required_columns)}")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Show example CSV format
    st.subheader("Example CSV Format")
    example_df = pd.DataFrame({
        'sepal_length': [5.1, 4.9, 4.7],
        'sepal_width': [3.5, 3.0, 3.2],
        'petal_length': [1.4, 1.4, 1.3],
        'petal_width': [0.2, 0.2, 0.2]
    })
    st.dataframe(example_df, use_container_width=True)

def main():
    st.set_page_config(page_title="Advanced ML Features", page_icon="🚀", layout="wide")
    
    st.title("🚀 Advanced ML Model Deployment Features")
    
    # Sidebar for feature selection
    st.sidebar.header("Advanced Features")
    feature = st.sidebar.selectbox(
        "Select Feature:",
        ["Customer Churn Analysis", "Model Comparison", "Batch Predictions"]
    )
    
    if feature == "Customer Churn Analysis":
        create_advanced_classification_example()
    elif feature == "Model Comparison":
        create_model_comparison()
    elif feature == "Batch Predictions":
        create_batch_prediction_interface()

if __name__ == "__main__":
    main()
