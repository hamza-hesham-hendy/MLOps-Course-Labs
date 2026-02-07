# Bank Customer Churn Prediction

A modular MLOps project for predicting bank customer churn using multiple machine learning models with MLflow tracking.

## Project Structure

MLOps-Course-Labs/
├── README.md  
├── requirements.txt  
├── dataset/  
│   └── Churn_Modelling.csv  
└── src/  
    ├── train.py           # Main training script with MLflow logging  
    ├── model_config.py    # (Optional / future extension)  

## Features

- Multi-Model Training: Train and compare multiple ML models in a single experiment  
- MLflow Integration: Full experiment tracking with metrics, parameters, and artifacts  
- Data Preprocessing Pipeline:
  - Class rebalancing using downsampling
  - Feature scaling with StandardScaler
  - One-hot encoding for categorical variables
- Automated Comparison: Generates a summary CSV with model performance rankings  

## Available Models

| Model | Description |
|------|------------|
| Logistic Regression | Linear classifier with regularization |
| Random Forest | Ensemble of decision trees |
| Decision Tree | Tree-based classifier |

## Installation

```bash
pip install -r requirements.txt
