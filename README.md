# Bank Customer Churn Prediction (MLOps)

This repository contains a modular Machine Learning pipeline designed to predict customer churn. It utilizes **MLflow** to track experiments, manage model artifacts, and compare the performance of different classification algorithms.

## 📂 Project Structure

```text
MLOps-Course-Labs/
├── artifacts/                # Local storage for plots and pickled transformers
│   ├── column_transformer.pkl
│   └── confusion_matrix_*.png
├── dataset/
│   └── Churn_Modelling.csv   # Raw customer data
├── src/
│   ├── train.py              # Single model training (Logistic Regression)
│   └── multimodel_train.py   # Multi-model experiment with nested runs
├── mlruns/                   # MLflow local database/tracking directory
├── model_comparison.csv      # Auto-generated CSV comparing all model metrics
├── requirements.txt          # Python dependencies
└── README.md

🚀 Features
Automated Preprocessing: Handles class imbalance via downsampling and uses ColumnTransformer for one-hot encoding and scaling.

Experiment Tracking: Automatically logs hyperparameters, accuracy, precision, recall, and F1-score to MLflow.

Model Signature: Includes input/output schemas in MLflow for easier deployment.

Nested Runs: multimodel_train.py uses nested MLflow runs to keep model comparisons organized under a single parent experiment.

🛠️ Installation
Clone the repository (or navigate to the folder).

Install dependencies:
pip install -r requirements.txt

📈 Usage
1. Start the Tracking Server
In a separate terminal, launch the MLflow UI:
mlflow ui --port 5000

It looks like you’re ready to get this project fully documented! I’ve put together a comprehensive README.md in raw Markdown format. It includes the correct file structure from your screenshots and explains how your two specific scripts (train.py and multimodel_train.py) operate within the MLflow ecosystem.

You can copy and paste the block below directly into your README.md file:

Markdown
# Bank Customer Churn Prediction (MLOps)

This repository contains a modular Machine Learning pipeline designed to predict customer churn. It utilizes **MLflow** to track experiments, manage model artifacts, and compare the performance of different classification algorithms.

## 📂 Project Structure

```text
MLOps-Course-Labs/
├── artifacts/                # Local storage for plots and pickled transformers
│   ├── column_transformer.pkl
│   └── confusion_matrix_*.png
├── dataset/
│   └── Churn_Modelling.csv   # Raw customer data
├── src/
│   ├── train.py              # Single model training (Logistic Regression)
│   └── multimodel_train.py   # Multi-model experiment with nested runs
├── mlruns/                   # MLflow local database/tracking directory
├── model_comparison.csv      # Auto-generated CSV comparing all model metrics
├── requirements.txt          # Python dependencies
└── README.md
🚀 Features
Automated Preprocessing: Handles class imbalance via downsampling and uses ColumnTransformer for one-hot encoding and scaling.

Experiment Tracking: Automatically logs hyperparameters, accuracy, precision, recall, and F1-score to MLflow.

Model Signature: Includes input/output schemas in MLflow for easier deployment.

Nested Runs: multimodel_train.py uses nested MLflow runs to keep model comparisons organized under a single parent experiment.

🛠️ Installation
Clone the repository (or navigate to the folder).

Install dependencies:

Bash
pip install -r requirements.txt
📈 Usage
1. Start the Tracking Server
In a separate terminal, launch the MLflow UI:

Bash
mlflow ui --port 5000
2. Run Experiments
To train the baseline Logistic Regression model:
python src/train.py

To run the full model comparison (Logistic Regression, Random Forest, Decision Tree):
python src/multimodel_train.py

It looks like you’re ready to get this project fully documented! I’ve put together a comprehensive README.md in raw Markdown format. It includes the correct file structure from your screenshots and explains how your two specific scripts (train.py and multimodel_train.py) operate within the MLflow ecosystem.

You can copy and paste the block below directly into your README.md file:

Markdown
# Bank Customer Churn Prediction (MLOps)

This repository contains a modular Machine Learning pipeline designed to predict customer churn. It utilizes **MLflow** to track experiments, manage model artifacts, and compare the performance of different classification algorithms.

## 📂 Project Structure

```text
MLOps-Course-Labs/
├── artifacts/                # Local storage for plots and pickled transformers
│   ├── column_transformer.pkl
│   └── confusion_matrix_*.png
├── dataset/
│   └── Churn_Modelling.csv   # Raw customer data
├── src/
│   ├── train.py              # Single model training (Logistic Regression)
│   └── multimodel_train.py   # Multi-model experiment with nested runs
├── mlruns/                   # MLflow local database/tracking directory
├── model_comparison.csv      # Auto-generated CSV comparing all model metrics
├── requirements.txt          # Python dependencies
└── README.md
🚀 Features
Automated Preprocessing: Handles class imbalance via downsampling and uses ColumnTransformer for one-hot encoding and scaling.

Experiment Tracking: Automatically logs hyperparameters, accuracy, precision, recall, and F1-score to MLflow.

Model Signature: Includes input/output schemas in MLflow for easier deployment.

Nested Runs: multimodel_train.py uses nested MLflow runs to keep model comparisons organized under a single parent experiment.

🛠️ Installation
Clone the repository (or navigate to the folder).

Install dependencies:

Bash
pip install -r requirements.txt
📈 Usage
1. Start the Tracking Server
In a separate terminal, launch the MLflow UI:

Bash
mlflow ui --port 5000
2. Run Experiments
To train the baseline Logistic Regression model:

Bash
python src/train.py
To run the full model comparison (Logistic Regression, Random Forest, Decision Tree):

Bash
python src/multimodel_train.py
3. Analyze Results
Open http://localhost:5000 to:

Compare F1-scores across different models.

View Confusion Matrix plots in the Artifacts section.

Download the column_transformer.pkl for inference.

📊 Models & Metrics
The pipeline tracks the following metrics for every run:

Accuracy: Overall correctness.

Precision/Recall: Performance on the "Churn" class.

F1-Score: Harmonic mean, used as the primary ranking metric in model_comparison.csv.

📦 Requirements
Python 3.8+

Scikit-learn

Pandas

MLflow

Matplotlib
