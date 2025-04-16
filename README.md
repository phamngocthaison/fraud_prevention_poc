# FRAUD DETECTION AND RISK ASSESSMENT SYSTEM

The system uses machine learning to detect fraudulent transactions and assess risk levels. This demo is designed for presentation in a seminar on "Fraud Detection/Fraud Rating System".

## Project Structure

The project is divided into independent modules for easy demonstration of each stage:

1. **Data Loading and Integration** (`1-data-loading.py`): Reads transaction data and fraud labels from JSON files
2. **Data Cleaning** (`2-data-cleaning.py`): Handles missing data, outliers, and data analysis
3. **Preprocessing and Feature Engineering** (`3-feature-engineering.py`): Creates new features and standardizes data
4. **Model Training** (`4-model-training.py`): Trains and evaluates fraud detection models
5. **Risk Rating System** (`5-risk-rating.py`): Builds a classification and risk level assessment system
6. **Demo Application** (`6-fraud-detection-app.py`): Visual interface for interacting with the system

## System Requirements

### Python Environment
- Python 3.7 or higher
- Required libraries listed in `requirements.txt`

### Installation

```bash
# Create virtual environment
python -m venv fraud-detection-venv

# Activate virtual environment
# Windows
fraud-detection-venv\Scripts\activate
# Linux/Mac
source fraud-detection-venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

## Running the Demo

### Running Individual Steps

You can run each step separately to demonstrate the detailed process:

```bash
# Step 1: Data Loading and Integration
python 1-data-loading.py

# Step 2: Data Cleaning
python 2-data-cleaning.py

# Step 3: Preprocessing and Feature Engineering
python 3-feature-engineering.py

# Step 4: Model Training
python 4-model-training.py

# Step 5: Risk Rating System
python 5-risk-rating.py

# Step 6: Run Demo Application
streamlit run 6-fraud-detection-app.py
```

### Running the Complete Process

To run all steps automatically, use the `run-all.py` script:

```bash
# Run complete process
python run-all.py

# Run only demo application (assuming previous steps completed)
python run-all.py --app

# Skip model training (use existing model)
python run-all.py --skip-training

# Start from specific step
python run-all.py --start-from 3

# Run only specific step
python run-all.py --only 4
```

## Data

The system uses datasets from Kaggle: [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)

### Data Structure

- `train_transaction.csv`: Transaction data
- `train_fraud_labels.json`: Fraud labels for transactions

### Output Directory

Analysis results and models are saved in the `output/` directory:

- `combined_data.csv`: Combined data
- `cleaned_data.csv`: Cleaned data
- `best_model.pkl`: Best fraud detection model
- `preprocessor.pkl`: Data preprocessor
- `risk_rating_system.pkl`: Risk rating system
- `fraud_risk_assessment_results.csv`: Risk assessment results
- Various charts and visualizations

## Key Features of the Demo Application

1. **Overview**: Introduction to the system and architecture
2. **Data Analysis**: Model performance evaluation through charts
3. **Sample Transactions**: View and analyze sample transactions with risk assessments
4. **Manual Assessment**: Input new transaction information and receive assessment results
5. **User Guide**: Detailed system usage instructions

## Risk Categories

The system classifies transactions into 4 risk categories:

1. **Low Risk (0-20)**: Allow automatic transaction processing
2. **Medium Risk (21-50)**: Require additional verification (OTP, biometrics)
3. **High Risk (51-80)**: Forward to staff for manual review
4. **Very High Risk (81-100)**: Suspend transaction and contact customer

## Transaction History Files

The demo application allows you to view sample transactions with risk classifications, helping to understand how the system works. Additionally, you can create and evaluate new transactions to test the system's fraud detection capabilities.

## Future Development

- Add deep learning models to detect complex fraud patterns
- Integrate behavioral analytics
- Build continuous learning system to update models with new data
- Develop API for integration with online payment systems

## Contact

For any questions or feedback, please contact:
- Email: phamngocthaison@gmail.com