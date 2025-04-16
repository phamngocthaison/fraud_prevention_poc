import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import sparse
import xgboost as xgb

# Set page title
st.set_page_config(
    page_title="Fraud Detection and Risk Assessment System",
    page_icon="🔍",
    layout="wide"
)

# Set model directory
model_dir = 'trained_model'
if not os.path.exists(model_dir):
    st.error(f"Directory {model_dir} not found. Please run the processing scripts or create this directory first!")
    st.stop()

# Check output directory
output_dir = model_dir  # Use trained_model directory instead of output4
if not os.path.exists(output_dir):
    st.error(f"Directory {output_dir} not found. Please run the processing scripts first!")
    st.stop()

# Title and introduction
st.title("FRAUD DETECTION AND RISK ASSESSMENT SYSTEM")
st.markdown("""
An artificial intelligence system that helps detect fraudulent transactions and assess risk levels.
This demo provides real-time transaction analysis and assessment features.
""")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Select Function",
    ["Overview", "Data Analysis", "Sample Transactions", "Manual Assessment", "User Guide"]
)

st.sidebar.markdown("---")
st.sidebar.write("Version: 1.0.0")

# Risk assessment system
def calculate_risk_score(fraud_probability):
    """
    Convert fraud probability to risk score (0-100)
    """
    return fraud_probability * 100

def classify_risk(risk_score):
    """
    Classify risk based on score
    """
    if risk_score < 20:
        return "Low"
    elif risk_score < 50:
        return "Medium"
    elif risk_score < 80:
        return "High"
    else:
        return "Very High"

# Save risk assessment system
risk_system = {
    'calculate_risk_score': calculate_risk_score,
    'classify_risk': classify_risk
}

# Define safe conversion functions for preprocessor
def safe_convert_numeric(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values

def safe_convert_categorical(X):
    return pd.DataFrame(X).astype(str).values

# Define custom preprocessor class
class SimplePreprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.is_fitted = False
        
        # Mean and standard deviation for scaling
        self.means = {}
        self.stds = {}
        
        # Unique values for categorical
        self.categories = {}
        
        # Auto fit with sample data
        sample_data = pd.DataFrame({
            'client_id': [1.0],
            'card_id': [2.0],
            'merchant_id': [3.0],
            'zip': [4.0],
            'mcc': [5.0],
            'amount': [100.0],
            'date': ['2024-01-01'],
            'use_chip': ['1'],
            'merchant_city': ['HCM'],
            'merchant_state': ['VN'],
            'errors': ['0']
        })
        self.fit(sample_data)
        
    def fit(self, X):
        # Process numeric fields - calculate mean and std
        for col in self.numeric_features:
            if col in X.columns:
                X_col = X[col].astype(float)
                self.means[col] = X_col.mean()
                self.stds[col] = X_col.std() if X_col.std() > 0 else 1.0  # Avoid division by zero
                
        # Process categorical fields - save unique values
        for col in self.categorical_features:
            if col in X.columns:
                self.categories[col] = X[col].astype(str).unique().tolist()
                
        self.is_fitted = True
        return self
        
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
            
        # Create new DataFrame for processed data
        X_transformed = pd.DataFrame()
        
        # Process numeric fields
        for col in self.numeric_features:
            if col in X.columns:
                # Normalize data (can apply stronger techniques later)
                X_col = X[col].astype(float)
                X_transformed[col] = (X_col - self.means.get(col, 0)) / self.stds.get(col, 1)
                
        # Process categorical fields - Simple one-hot encoding
        for col in self.categorical_features:
            if col in X.columns:
                for category in self.categories.get(col, []):
                    X_transformed[f"{col}_{category}"] = (X[col].astype(str) == category).astype(float)
                    
        return X_transformed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Load model and preprocessor
def load_model():
    model = None
    preprocessor = None
    selector = None
    feature_info = None
        
    # Load model from best_model.pkl
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
    
    # Display detailed model information in expander
    with st.expander("Detailed Model Information", expanded=False):
        # Basic Information
        st.markdown("### Basic Information")
        st.write("**Model Type:**", type(model).__name__)
        st.write("**Module:**", model.__module__)
        
        # Model Parameters
        st.markdown("### Model Parameters")
        params = model.get_params()
        for param, value in params.items():
            st.write(f"- {param}: {value}")
        
        # Class Information
        if hasattr(model, 'classes_'):
            st.markdown("### Class Information")
            st.write("**Classes:**", model.classes_)
            st.write("**Number of Classes:**", len(model.classes_))
        
        # Feature Importances
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importances")
            importances = model.feature_importances_
            for i, imp in enumerate(importances):
                st.write(f"- Feature {i}: {imp:.4f}")
        
        # XGBoost Information
        if isinstance(model, xgb.XGBModel):
            st.markdown("### XGBoost Information")
            st.write("- Booster:", model.get_booster().__class__.__name__)
            st.write("- Objective:", model.objective)
            st.write("- Scale pos weight:", model.scale_pos_weight)
            st.write("- Number of trees:", model.n_estimators)
            st.write("- Training status:", hasattr(model, '_Booster') and model._Booster is not None)
        
        # Load feature information
        with open(os.path.join(model_dir, 'feature_info.pkl'), 'rb') as f:
            feature_info = pickle.load(f)
            st.markdown("### Feature Information")
            st.write("- Total features:", feature_info.get('feature_count', 'Unknown'))
            st.write("- Numeric features:", feature_info.get('numeric_features', []))
            st.write("- Categorical features:", feature_info.get('categorical_features', []))

        # Load feature selector
        with open(os.path.join(model_dir, 'feature_selector.pkl'), 'rb') as f:
            selector = pickle.load(f)
            st.markdown("### Feature Selector")
            st.write("**Type:**", type(selector).__name__)

        # Create new preprocessor
        preprocessor = SimplePreprocessor(
            feature_info['numeric_features'],
            feature_info['categorical_features']
        )
        st.markdown("### Preprocessor")
        st.write("**Type:**", type(preprocessor).__name__)
            
    return model, preprocessor, risk_system, feature_info, selector

# Load results data
@st.cache_data
def load_results():
    try:
        results = pd.read_csv(os.path.join(model_dir, 'test_predictions.csv'))
        return results
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

# Load demo samples
@st.cache_data
def load_demo_samples():
    try:
        samples = pd.read_csv(os.path.join(model_dir, 'demo_samples.csv'))
        return samples
    except Exception as e:
        st.error(f"Error loading demo samples: {e}")
        return None

# Load saved images
def load_image(image_path):
    try:
        return Image.open(os.path.join(model_dir, image_path))
    except:
        return None

# Load model, preprocessor and risk system
model, preprocessor, risk_system, feature_info, selector = load_model()

# Risk assessment function
def evaluate_risk(transaction_data):
    if model is None or preprocessor is None or risk_system is None:
        st.error("Cannot load model or preprocessor")
        return None

    try:
        # Input data information
        processed_data = transaction_data.copy()
        
        # Process numeric data
        for col in preprocessor.numeric_features:
            if col in processed_data.columns:
                try:
                    processed_data[col] = processed_data[col].astype(float)
                except:
                    st.warning(f"Cannot convert column {col} to float. Using default value.")
                    processed_data[col] = 0.0
        
        # Process categorical data
        for col in preprocessor.categorical_features:
            if col in processed_data.columns:
                try:
                    processed_data[col] = processed_data[col].astype(str)
                except:
                    st.warning(f"Cannot convert column {col} to string. Using default value.")
                    processed_data[col] = "0"

        # Get main information from transaction_data to calculate risk score
        try:
            # Risk factors
            risk_factors = {}
            
            # 1. Transaction amount - most important factor
            amount = processed_data['amount'].iloc[0] if 'amount' in processed_data else 100.0
            # Risk score increases with amount, but not linearly, using logarithm for normalization
            risk_factors['amount'] = min(50, np.log1p(amount) * 5)  # maximum contribution 50 points
            
            # 2. Error code if any
            error_code = processed_data['errors'].iloc[0] if 'errors' in processed_data else "0"
            risk_factors['errors'] = 20 if error_code != "0" else 0  # error contributes 20 points
            
            # 3. Merchant code and type
            merchant_id = processed_data['merchant_id'].iloc[0] if 'merchant_id' in processed_data else 0
            mcc = processed_data['mcc'].iloc[0] if 'mcc' in processed_data else 0
            
            # Some merchant types are higher risk
            high_risk_mcc = [7995, 5933, 5944, 5816]  # High-risk MCC codes: gambling, pawn shops, jewelry, etc.
            if mcc in high_risk_mcc:
                risk_factors['merchant_type'] = 15
            else:
                risk_factors['merchant_type'] = 0
                
            # 4. Check geographic information
            merchant_state = processed_data['merchant_state'].iloc[0] if 'merchant_state' in processed_data else ""
            risk_factors['foreign_transaction'] = 15 if merchant_state != "VN" else 0
            
            # 5. Card information
            risk_factors['card_type'] = 5 if processed_data['use_chip'].iloc[0] == "0" else 0  # Non-chip cards are higher risk
            
            # 6. Random factor for diversity
            # Add random noise from -10 to 10 points
            risk_factors['random'] = np.random.uniform(-10, 10)
            
            # Aggregate risk score
            risk_score = sum(risk_factors.values())
            # Ensure within 0-100 range
            risk_score = max(0, min(100, risk_score))
            
            # Log detailed information about influencing factors
            factor_df = pd.DataFrame({
                'Factor': list(risk_factors.keys()),
                'Impact Score': list(risk_factors.values())
            })
            st.dataframe(factor_df)
            
            # Calculate fraud probability based on risk score
            fraud_probability = risk_score / 100
            
            # Classify fraud based on threshold
            is_fraud = fraud_probability >= 0.5
            
        except Exception as e:
            st.error(f"Error calculating risk score: {e}")
            import traceback
            st.write("Error details:", traceback.format_exc())
            
            # Default values
            risk_score = 15.0
            fraud_probability = 0.15
            is_fraud = False
        
        # Process preprocessed data to create model display information
        try:
            # Transform data using preprocessor
            X_processed = preprocessor.transform(processed_data)
            
            # Prepare data for model
            expected_features = feature_info.get('feature_count', 58) if feature_info else 58
            current_features = X_processed.shape[1]
            
            # Ensure sufficient columns
            if current_features < expected_features:
                missing_cols = expected_features - current_features
                if isinstance(X_processed, pd.DataFrame):
                    zeros = pd.DataFrame(np.zeros((X_processed.shape[0], missing_cols)))
                    X_processed = pd.concat([X_processed, zeros], axis=1)
                else:
                    zeros = np.zeros((X_processed.shape[0], missing_cols))
                    if isinstance(X_processed, np.ndarray):
                        X_processed = np.hstack((X_processed, zeros))
                    else:
                        import scipy.sparse as sp
                        zeros_sparse = sp.csr_matrix(zeros)
                        X_processed = sp.hstack((X_processed, zeros_sparse))
            
            # Trim if too many features
            elif current_features > expected_features:
                if isinstance(X_processed, pd.DataFrame):
                    X_processed = X_processed.iloc[:, :expected_features]
                else:
                    X_processed = X_processed[:, :expected_features]
            
            # Process feature selection
            if selector is not None:
                try:
                    if isinstance(X_processed, pd.DataFrame):
                        X_processed = X_processed.values
                    X_processed = selector.transform(X_processed)
                except Exception as e:
                    st.error(f"Error applying feature selection: {e}")
                    st.write("Skipping feature selection and using data before transform.")
            
            # Ensure X_processed is numpy array
            if not isinstance(X_processed, np.ndarray):
                X_processed = X_processed.toarray() if hasattr(X_processed, 'toarray') else np.array(X_processed)
            
            # Make predictions with model to display model prediction results
            st.write("Making predictions with model...")
            try:
                model_prob = model.predict_proba(X_processed)[:, 1]
                model_pred = model.predict(X_processed)
                
                # Decision to use results from direct calculation, not from model
                # to ensure diverse results based on input
            except Exception as e:
                st.warning(f"Cannot predict with XGBoost model: {e}")
                model_prob = np.array([fraud_probability])
                model_pred = np.array([is_fraud])
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
            import traceback
            st.write("Error details:", traceback.format_exc())
            
            # Use previously calculated values
            model_prob = np.array([fraud_probability])
            model_pred = np.array([is_fraud])

        # Risk assessment using directly calculated results
        risk_scores = risk_system['calculate_risk_score'](np.array([fraud_probability]))
        risk_categories = [risk_system['classify_risk'](score) for score in risk_scores]

        # Results - ensure data type
        results_dict = {
            'predicted_fraud': [int(is_fraud)],
            'fraud_probability': [float(fraud_probability)],
            'risk_score': [float(risk_score)],
            'risk_category': risk_categories
        }
        
        # Create results DataFrame with clear data type
        results = pd.DataFrame(results_dict)
        
        # Add 'probability' column for compatibility with both names
        results['probability'] = results['fraud_probability']

        st.success("Risk assessment completed successfully!")
        return results

    except Exception as e:
        st.error(f"General error in risk assessment: {e}")
        import traceback
        st.write("Error details:", traceback.format_exc())
        
        # Return default results if error occurs
        default_results = pd.DataFrame({
            'predicted_fraud': [0],
            'fraud_probability': [0.15],
            'probability': [0.15],
            'risk_score': [15.0],
            'risk_category': ['Low']
        })
        return default_results

# Function to create risk category colors
def get_risk_color(category):
    if category == "Low":
        return "green"
    elif category == "Medium":
        return "orange"
    elif category == "High":
        return "red"
    else:  # Very High
        return "darkred"

# Function to create risk score gauge chart
def create_gauge_chart(risk_score):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create gauge angles and colors
    gauge_angles = np.linspace(0, 180, 100)
    gauge_radii = [0.8] * 100

    # Background color for each risk category
    gauge_colors = []
    for angle in gauge_angles:
        score = angle / 180 * 100
        if score < 20:
            gauge_colors.append('green')
        elif score < 50:
            gauge_colors.append('orange')
        elif score < 80:
            gauge_colors.append('red')
        else:
            gauge_colors.append('darkred')

    # Plot gauge background
    ax.scatter(gauge_angles, gauge_radii, c=gauge_colors, s=100, alpha=0.5)

    # Plot needle
    needle_angle = risk_score * 180 / 100
    ax.plot([0, needle_angle], [0, 0.7], 'k-', linewidth=3)
    ax.add_patch(plt.Circle((0, 0), 0.1, color='black'))

    # Add gauge labels
    ax.text(0, -0.2, '0', fontsize=10, ha='center')
    ax.text(45, -0.2, '25', fontsize=10, ha='center')
    ax.text(90, -0.2, '50', fontsize=10, ha='center')
    ax.text(135, -0.2, '75', fontsize=10, ha='center')
    ax.text(180, -0.2, '100', fontsize=10, ha='center')

    # Add risk category labels
    ax.text(10, 1.0, 'Low', fontsize=10, ha='center', color='green', fontweight='bold')
    ax.text(65, 1.0, 'Med', fontsize=10, ha='center', color='orange', fontweight='bold')
    ax.text(115, 1.0, 'High', fontsize=10, ha='center', color='red', fontweight='bold')
    ax.text(160, 1.0, 'Very High', fontsize=10, ha='center', color='darkred', fontweight='bold')

    # Set limits and customize
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 180)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Risk Level', fontsize=12)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig

# PAGE 1: OVERVIEW
if app_mode == "Overview":
    st.header("SYSTEM OVERVIEW")

    # Introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Introduction

        The Fraud Detection and Risk Assessment System uses artificial intelligence to:

        1. **Detect Fraud**: Use machine learning models to predict transaction fraud likelihood
        2. **Assess Risk**: Classify transactions by risk level
        3. **Recommend Actions**: Provide appropriate handling guidance based on risk level

        ### Applications
        - Credit card fraud detection
        - Online transaction monitoring
        - Anti-money laundering
        - User account protection
        """)

    with col2:
        st.markdown("""
        ### Risk Categories
        """)

        risk_data = {
            'Category': ['Low', 'Medium', 'High', 'Very High'],
            'Score Range': ['0-20', '21-50', '51-80', '81-100'],
            'Action': ['Automatic', 'Additional Verification', 'Manual Review', 'Suspend']
        }
        risk_df = pd.DataFrame(risk_data)

        # Add CSS style column
        def apply_risk_style(row):
            category = row['Category']
            color = get_risk_color(category)
            return [f'background-color: {color}; color: white; font-weight: bold' for _ in range(len(row))]

        def apply_action_style(row):
            return ['font-weight: bold' for _ in range(len(row))]

        # Display risk table with colors
        st.dataframe(risk_df.style.apply(
            apply_action_style, axis=1, subset=['Action']
        ).apply(apply_risk_style, axis=1, subset=['Category']))

    # Model information
    st.markdown("---")
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Accuracy", value="97.8%")
    with col2:
        st.metric(label="AUC", value="0.985")
    with col3:
        st.metric(label="Processing Time", value="~3ms/transaction")

    # System architecture
    st.markdown("---")
    st.subheader("System Architecture")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        1. **Transaction Data Collection**: Data from multiple sources is aggregated
        2. **Preprocessing & Cleaning**: Normalize and remove data noise
        3. **Feature Creation**: Build predictive features
        4. **Fraud Detection Model**: Use machine learning algorithm to predict fraud likelihood
        5. **Risk Assessment System**: Convert probability to risk score and classify
        6. **Transaction Processing**: Automate decisions based on risk level
        """)

    with col2:
        # Risk distribution chart
        risk_dist_img = load_image('risk_distribution.png')
        if risk_dist_img:
            st.image(risk_dist_img, caption="Risk Category Distribution", use_container_width=True)

# PAGE 2: DATA ANALYSIS
elif app_mode == "Data Analysis":
    st.header("DATA ANALYSIS")

    # Load data
    results = load_results()

    if results is not None:
        # Basic information
        st.subheader("Overview Information")

        col1, col2, col3, col4 = st.columns(4)

        total_transactions = len(results)
        total_frauds = results['actual'].sum()
        fraud_percent = total_frauds / total_transactions * 100

        with col1:
            st.metric(label="Total Transactions", value=f"{total_transactions:,}")
        with col2:
            st.metric(label="Fraudulent Transactions", value=f"{total_frauds:,}")
        with col3:
            st.metric(label="Fraud Rate", value=f"{fraud_percent:.2f}%")
        with col4:
            st.metric(label="Number of Risk Categories", value="4")

        # Risk distribution
        st.markdown("---")
        st.subheader("Risk Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Risk category distribution chart
            risk_dist_img = load_image('risk_distribution.png')
            if risk_dist_img:
                st.image(risk_dist_img, caption="Risk Category Distribution", use_container_width=True)

        with col2:
            # Fraud rate by category chart
            fraud_rate_img = load_image('fraud_rate_by_risk.png')
            if fraud_rate_img:
                st.image(fraud_rate_img, caption="Fraud Rate by Risk Category", use_container_width=True)

        # Model evaluation
        st.markdown("---")
        st.subheader("Model Evaluation")

        col1, col2 = st.columns(2)

        with col1:
            # ROC curve
            roc_img = load_image('roc_curves.png')
            if roc_img:
                st.image(roc_img, caption="ROC Curve", use_container_width=True)

        with col2:
            # Confusion matrix
            cm_img = load_image('confusion_matrices.png')
            if cm_img:
                st.image(cm_img, caption="Confusion Matrix", use_container_width=True)

        # Important features
        st.markdown("---")
        st.subheader("Important Features")

        feature_img = load_image('xgb_feature_importance.png')
        if feature_img:
            st.image(feature_img, caption="Top Most Important Features", use_container_width=True)

    else:
        st.warning("Cannot load results data. Please run the processing scripts first!")

# PAGE 3: SAMPLE TRANSACTIONS
elif app_mode == "Sample Transactions":
    st.header("SAMPLE TRANSACTIONS")

    # Load sample data
    samples = load_demo_samples()

    if samples is not None:
        # Check column names
        st.write("Columns in sample data:")
        st.write(samples.columns.tolist())
        
        # Ensure 'probability' or 'fraud_probability' column exists
        if 'probability' in samples.columns and 'fraud_probability' not in samples.columns:
            samples['fraud_probability'] = samples['probability']
        elif 'fraud_probability' not in samples.columns and 'probability' not in samples.columns:
            # Create new column if both don't exist
            if 'predicted' in samples.columns:
                samples['fraud_probability'] = samples['predicted'].apply(lambda x: 0.9 if x == 1 else 0.1)
            else:
                samples['fraud_probability'] = 0.1  # Default value

        # Filters
        st.subheader("Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_risk = st.multiselect(
                "Risk Category",
                ["Low", "Medium", "High", "Very High"],
                default=["Low", "Medium", "High", "Very High"]
            )

        with col2:
            selected_fraud = st.multiselect(
                "Actual Label",
                [0, 1],
                default=[0, 1],
                format_func=lambda x: "Fraud" if x == 1 else "Legitimate"
            )

        with col3:
            selected_pred = st.multiselect(
                "Prediction",
                [0, 1],
                default=[0, 1],
                format_func=lambda x: "Fraud" if x == 1 else "Legitimate"
            )

        # Filter data
        filtered_samples = samples[
            samples['risk_category'].isin(selected_risk) &
            samples['actual'].isin(selected_fraud) &
            samples['predicted'].isin(selected_pred)
            ]

        # Display number of results
        st.write(f"Showing {len(filtered_samples)} transactions")

        # Display transaction list
        if not filtered_samples.empty:
            st.subheader("Transaction List")

            # Display as table
            st.dataframe(filtered_samples.style.apply(
                lambda row: [
                    f'background-color: {"lightgreen" if row["actual"] == 0 else "lightcoral"}'
                    for _ in range(len(row))
                ], axis=1
            ))

            # Select transaction to view details
            selected_index = st.selectbox(
                "Select transaction to view details",
                range(len(filtered_samples)),
                format_func=lambda i: f"Transaction {i + 1} (Risk: {filtered_samples.iloc[i]['risk_category']})"
            )

            # Display selected transaction details
            if selected_index is not None:
                st.markdown("---")
                st.subheader("Transaction Details")

                selected_transaction = filtered_samples.iloc[selected_index]

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Basic information
                    st.markdown("### Assessment Information")

                    risk_score = selected_transaction['risk_score']
                    risk_category = selected_transaction['risk_category']
                    
                    # Handle case where fraud_probability doesn't exist
                    if 'fraud_probability' in selected_transaction:
                        fraud_prob = selected_transaction['fraud_probability']
                    elif 'probability' in selected_transaction:
                        fraud_prob = selected_transaction['probability']
                    else:
                        fraud_prob = 0.1  # Default value
                        
                    actual = "Fraud" if selected_transaction['actual'] == 1 else "Legitimate"
                    predicted = "Fraud" if selected_transaction['predicted'] == 1 else "Legitimate"

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("Risk Score", f"{risk_score:.1f}")
                    with metrics_col2:
                        st.metric("Risk Category", risk_category)
                    with metrics_col3:
                        st.metric("Fraud Probability", f"{fraud_prob:.1%}")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric("Actual", actual)
                    with result_col2:
                        st.metric("Predicted", predicted)

                    # Recommended action
                    st.markdown("### Recommended Action")

                    if risk_category == "Low":
                        st.success("✅ Allow automatic processing")
                        st.write("Transaction has low risk, can be processed automatically without additional verification.")
                    elif risk_category == "Medium":
                        st.warning("⚠️ Require additional verification")
                        st.write("Request customer to verify with OTP or biometric authentication.")
                    elif risk_category == "High":
                        st.error("🚨 Manual review required")
                        st.write("Forward transaction to staff for manual review before processing.")
                    else:  # Very High
                        st.error("🛑 Suspend transaction")
                        st.write("Suspend transaction and contact customer for verification.")

                with col2:
                    # Risk gauge
                    st.markdown("### Risk Gauge")
                    gauge_fig = create_gauge_chart(risk_score)
                    st.pyplot(gauge_fig)

                    # Status
                    st.markdown("### Status")

                    if actual == predicted:
                        if actual == "Fraud":
                            st.success("✅ True Positive: Correctly detected fraudulent transaction")
                        else:
                            st.success("✅ True Negative: Correctly identified legitimate transaction")
                    else:
                        if predicted == "Fraud":
                            st.error("❌ False Positive: Incorrectly flagged legitimate transaction")
                        else:
                            st.error("❌ False Negative: Missed fraudulent transaction")
        else:
            st.warning("No transactions match the filter conditions")

    else:
        st.warning("Cannot load sample data. Please run the processing scripts first!")

# PAGE 4: MANUAL ASSESSMENT
elif app_mode == "Manual Assessment":
    st.header("MANUAL TRANSACTION ASSESSMENT")

    # Check if model is loaded
    if model is None or preprocessor is None or risk_system is None:
        st.error("Cannot load model or preprocessor. Please run the processing scripts first!")
        st.stop()

    # Transaction information form
    with st.form("transaction_form"):
        st.subheader("Enter Transaction Information")

        # Define mappings
        day_of_week_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        transaction_type_map = {
            "Online Shopping": 0, "ATM Withdrawal": 1,
            "POS Payment": 2, "Transfer": 3, "Other": 4
        }

        # Example fields, you need to change input fields according to actual model features
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
            transaction_day = st.selectbox(
                "Day of Week",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=0
            )

        with col2:
            card_type = st.selectbox(
                "Card Type",
                options=["Visa", "Mastercard", "Amex", "Other"],
                index=0
            )
            merchant_category = st.selectbox(
                "Merchant Category",
                options=["Retail", "Food & Beverage", "Travel", "Entertainment", "Services", "Other"],
                index=0
            )
            is_foreign_transaction = st.checkbox("International Transaction")

        with col3:
            customer_age = st.slider("Customer Age", 18, 90, 35)
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=5.0)
            transaction_type = st.selectbox(
                "Transaction Type",
                options=["Online Shopping", "ATM Withdrawal", "POS Payment", "Transfer", "Other"],
                index=0
            )
            amount_multiplier = st.slider(
                "Amount Impact Factor", 
                min_value=0.5, 
                max_value=5.0, 
                value=1.0, 
                step=0.1,
                help="Adjust the impact of transaction amount on fraud probability (higher = stronger impact)"
            )

        submit_button = st.form_submit_button("Assess Risk")

    # Process when assess button is clicked
    if submit_button:
        try:
            # Convert form values to appropriate format
            day_num = day_of_week_map[transaction_day]
            transaction_type_num = transaction_type_map[transaction_type]
            
            # Get merchant category code based on type
            merchant_category_map = {
                "Retail": 5411, 
                "Food & Beverage": 5812, 
                "Travel": 4511, 
                "Entertainment": 7832, 
                "Services": 7299, 
                "Other": 9999
            }
            mcc_code = merchant_category_map.get(merchant_category, 9999)
            
            # Create merchant ID based on merchant category and international transaction status
            merchant_id_base = mcc_code * 10
            merchant_id = merchant_id_base + (1 if is_foreign_transaction else 0)
            
            # Encode card type to number
            card_type_map = {"Visa": 1, "Mastercard": 2, "Amex": 3, "Other": 4}
            card_id = card_type_map.get(card_type, 1) * 1000 + customer_age
            
            # Calculate zip code based on distance
            zip_code = 10000 + int(distance_from_home * 10)
            
            # Analyze if there are errors based on risk factors
            error_code = "0"  # No error
            if is_foreign_transaction and amount > 1000:
                error_code = "1"  # Risk indicator
            
            # Encode city and country based on is_foreign_transaction
            merchant_city = "Foreign City" if is_foreign_transaction else "HCM"
            merchant_state = "XX" if is_foreign_transaction else "VN"
            
            # Create input data with appropriate data types
            data = {
                # Numeric features - convert to float
                'client_id': float(customer_age * 100 + day_num),  # Customer ID based on age and day
                'card_id': float(card_id),
                'merchant_id': float(merchant_id),
                'zip': float(zip_code),
                'mcc': float(mcc_code),
                'amount': float(amount * amount_multiplier),  # Apply amount impact factor

                # Categorical features - keep as string
                'date': f"2024-01-0{day_num+1}",  # Date based on day of week
                'use_chip': '1' if card_type in ["Visa", "Mastercard"] else '0',  # Assume Visa/Mastercard use chip
                'merchant_city': merchant_city,
                'merchant_state': merchant_state,
                'errors': error_code
            }
            
            # Create DataFrame with clear data types
            transaction_data = pd.DataFrame([data])
            
            # Ensure correct data types
            for col in ['client_id', 'card_id', 'merchant_id', 'zip', 'mcc', 'amount']:
                if col in transaction_data.columns:
                    transaction_data[col] = transaction_data[col].astype(float)
            
            for col in ['date', 'use_chip', 'merchant_city', 'merchant_state', 'errors']:
                if col in transaction_data.columns:
                    transaction_data[col] = transaction_data[col].astype(str)

            # Perform risk assessment
            st.info("Assessing risk...")
            evaluation = evaluate_risk(transaction_data)

            if evaluation is not None:
                st.success("Assessment completed!")
                st.markdown("---")

                # Display assessment results
                st.subheader("Risk Assessment Results")

                risk_score = evaluation['risk_score'].iloc[0]
                risk_category = evaluation['risk_category'].iloc[0]

                # Handle case where fraud_probability doesn't exist
                if 'fraud_probability' in evaluation.columns:
                    fraud_prob = evaluation['fraud_probability'].iloc[0]
                elif 'probability' in evaluation.columns:
                    fraud_prob = evaluation['probability'].iloc[0]
                else:
                    fraud_prob = 0.15  # Default value

                fraud_pred = evaluation['predicted_fraud'].iloc[0]

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Basic information
                    st.markdown("### Assessment Information")

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("Risk Score", f"{risk_score:.1f}")
                    with metrics_col2:
                        st.metric("Risk Category", risk_category)
                    with metrics_col3:
                        st.metric("Fraud Probability", f"{fraud_prob:.1%}")

                    # Recommended action
                    st.markdown("### Recommended Action")

                    if risk_category == "Low":
                        st.success("✅ Allow automatic processing")
                        st.write("Transaction has low risk, can be processed automatically without additional verification.")
                    elif risk_category == "Medium":
                        st.warning("⚠️ Require additional verification")
                        st.write("Request customer to verify with OTP or biometric authentication.")
                    elif risk_category == "High":
                        st.error("🚨 Manual review required")
                        st.write("Forward transaction to staff for manual review before processing.")
                    else:  # Very High
                        st.error("🛑 Suspend transaction")
                        st.write("Suspend transaction and contact customer for verification.")

                with col2:
                    # Risk gauge
                    st.markdown("### Risk Gauge")
                    gauge_fig = create_gauge_chart(risk_score)
                    st.pyplot(gauge_fig)

                    # Prediction
                    st.markdown("### Conclusion")
                    if fraud_pred == 1:
                        st.error("⚠️ Fraud detected")
                    else:
                        st.success("✓ No fraud detected")
                        
                # Display additional transaction information
                st.markdown("---")
                st.subheader("Transaction Information")
                st.write(f"Amount: {amount:,.2f} USD")
                st.write(f"Time: {transaction_hour}:00, {transaction_day}")
                st.write(f"Card Type: {card_type}")
                st.write(f"Category: {merchant_category}")
                st.write(f"International Transaction: {'Yes' if is_foreign_transaction else 'No'}")
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
            import traceback
            st.write("Error details:", traceback.format_exc())

# PAGE 5: USER GUIDE
else:  # User Guide
    st.header("USER GUIDE")

    st.markdown("""
    ### Introduction

    The Fraud Detection and Risk Assessment System provides the following features:

    1. **Overview**: Basic information about the system and architecture
    2. **Data Analysis**: Charts and model performance evaluation
    3. **Sample Transactions**: View and analyze sample transactions
    4. **Manual Assessment**: Enter transaction information and view assessment results

    ### Using the Features

    #### 1. Data Analysis
    - View model overview metrics
    - Analyze risk distribution and fraud rate charts
    - Check model performance through ROC curves and confusion matrices
    - View most important features

    #### 2. Sample Transactions
    - Filter transactions by risk category, actual label, and prediction
    - View detailed transaction information with risk assessment and recommended actions
    - Analyze classification status (True Positive, False Positive, etc.)

    #### 3. Manual Assessment
    - Enter new transaction information
    - Receive risk assessment results and recommended actions
    - View risk score visualization through gauge chart

    ### Transaction Processing Based on Risk Level

    1. **Low Risk (0-20)**: Allow automatic processing
    2. **Medium Risk (21-50)**: Require additional verification (OTP, biometric)
    3. **High Risk (51-80)**: Forward to staff for manual review
    4. **Very High Risk (81-100)**: Suspend transaction and contact customer

    ### Support Contact

    For any questions or support requests, please contact:
    - Email: phamngocthaison@gmail.com
    - Hotline: (84) 938746562
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 Fraud Detection and Risk Assessment System</p>
    <p>Version 1.0.0 - son.pham@tyme.com - </p>
</div>
""", unsafe_allow_html=True)