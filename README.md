Bank Customer Churn Prediction

A modular MLOps project for predicting bank customer churn using multiple machine learning models with MLflow tracking.

Project Structure

MLOps-Course-Labs/
├── README.md
├── requirements.txt
├── dataset/
│ └── Churn_Modelling.csv
└── src/
    ├── train.py # Main training script with MLflow logging
    ├── model_config.py # (Optional / future extension)

Features

Multi-Model Training: Train and compare multiple ML models in a single experiment

MLflow Integration: Full experiment tracking with metrics, parameters, and artifacts

Data Preprocessing Pipeline:

Class rebalancing using downsampling

Feature scaling with StandardScaler

One-hot encoding for categorical variables

Automated Comparison: Generates a summary CSV with model performance rankings

Available Models
Model	Description
Logistic Regression	Linear classifier with regularization
Random Forest	Ensemble of decision trees
Decision Tree	Tree-based classifier
Installation
pip install -r requirements.txt

Usage
1. Start MLflow Server
mlflow ui --port 5000

2. Run Training
cd src
python train.py

3. View Results

Open the following URL in your browser:

http://localhost:5000

Adding New Models

New models can be added directly in train.py by:

Creating a new training function

Adding the model to the models list

Logging parameters and metrics through MLflow

This design allows easy extension without modifying the core pipeline.

Metrics Tracked

Accuracy: Overall prediction accuracy

Precision: Positive predictive value

Recall: True positive rate

F1 Score: Harmonic mean of precision and recall

Artifacts

Each model run logs:

Trained model (MLflow sklearn format)

Column transformer (column_transformer.pkl)

Confusion matrix plot (.png)

Additionally, the parent run logs:

model_comparison.csv summarizing all models sorted by F1 score

Configuration

Model logic and hyperparameters are defined in src/train.py.
You can modify this file to:

Adjust hyperparameters

Enable or disable models

Add new classifiers

Requirements

Python 3.8+

scikit-learn

pandas

matplotlib

mlflow

joblib

License

MIT