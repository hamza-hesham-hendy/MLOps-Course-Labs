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

Model Signature: Includes input/output schemas in MLflow for easier deployment and validation.

Nested Runs: multimodel_train.py uses nested MLflow runs to keep model comparisons organized under a single parent experiment.


🛠️ Installation
1.Clone the repository (or navigate to your project folder).
2.Install dependencies:
pip install -r requirements.txt

📈 Usage
1. Start the Tracking Server
In a separate terminal, launch the MLflow UI to monitor your experiments:
mlflow ui --port 5000

Yes, the structure is logically sound, but the formatting within the README block you provided became a bit messy because some headers weren't properly formatted and the descriptive text (the "AI commentary") was accidentally mixed into the Markdown code.

To make it actually ready for a copy-paste into your README.md file, I have cleaned it up, added proper Markdown syntax for the lists, and ensured the bash commands are in the correct code blocks.

📝 Final README.md (Copy & Paste this)
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

Model Signature: Includes input/output schemas in MLflow for easier deployment and validation.

Nested Runs: multimodel_train.py uses nested MLflow runs to keep model comparisons organized under a single parent experiment.

🛠️ Installation
Clone the repository (or navigate to your project folder).

Install dependencies:

Bash
pip install -r requirements.txt
📈 Usage
1. Start the Tracking Server
In a separate terminal, launch the MLflow UI to monitor your experiments:

Bash
mlflow ui --port 5000
2. Run Experiments
To train the baseline Logistic Regression model:
python src/train.py

To run the full model comparison (Logistic Regression, Random Forest, Decision Tree):
python src/multimodel_train.py

3. Analyze Results
Open http://localhost:5000 in your browser to:

Compare F1-scores across different model types.

View Confusion Matrix plots stored in the Artifacts section.

Download the column_transformer.pkl for use in production inference.

📊 Models & Metrics
The pipeline tracks the following metrics for every run:

Accuracy: Overall correctness of the model.

Precision/Recall: Performance metrics specifically for the "Churn" (positive) class.

F1-Score: Harmonic mean of precision and recall, used as the primary ranking metric in model_comparison.csv.

📦 Requirements
Python 3.8+

Scikit-learn

Pandas

MLflow

Matplotlib

