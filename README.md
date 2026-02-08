# 🏦 Bank Customer Churn Prediction (MLOps)
A modular **end-to-end MLOps machine learning pipeline** for predicting bank customer churn.  
The project leverages **MLflow** for experiment tracking, model comparison, artifact logging, and reproducibility.

---

## 📌 Overview

This repository demonstrates best practices in applied MLOps, including:
- Automated preprocessing
- Multiple model experimentation
- Centralized experiment tracking
- Reproducible training pipelines
- Metric-driven model selection

- ---
The primary goal is to **compare multiple classifiers** and identify the best churn prediction model using **F1-score**.
- ---

## 📂 Project Structure
````md
MLOps-Course-Labs/
├── artifacts/                # Saved plots and serialized transformers
│   ├── column_transformer.pkl
│   └── confusion_matrix_*.png
├── dataset/
│   └── Churn_Modelling.csv   # Raw customer data
├── src/
│   ├── train.py              # Baseline single-model training
│   └── multimodel_train.py   # Multi-model training with nested MLflow runs
├── mlruns/                   # Local MLflow tracking directory
├── model_comparison.csv      # Auto-generated model ranking (by F1-score)
├── requirements.txt          # Project dependencies
└── README.md
````

---

## ✨ Key Features

### 🔄 Automated Preprocessing

* Handles class imbalance via **downsampling**
* Uses `ColumnTransformer` for:

  * Numerical feature scaling
  * Categorical feature one-hot encoding

### 📊 Experiment Tracking with MLflow

* Logs:

  * Model parameters
  * Accuracy, Precision, Recall, and F1-score
* Stores:

  * Confusion matrix plots
  * Preprocessing pipelines
  * Model signatures (input/output schema)

### 🧪 Multi-Model Experiments

* `multimodel_train.py` runs:

  * Logistic Regression
  * Random Forest
  * Decision Tree
* Uses **nested MLflow runs** under a single experiment

### 📈 Automated Model Comparison

* Generates `model_comparison.csv`
* Ranks all models by **F1-score** (primary metric)

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Start the MLflow Tracking Server

```bash
mlflow ui --port 5000
```

Then open:

```
http://localhost:5000
```

---

### 2️⃣ Run Model Training

Train the baseline Logistic Regression model:

```bash
python src/train.py
```

Run the full multi-model experiment:

```bash
python src/multimodel_train.py
```

---

### 3️⃣ Analyze Results

From the MLflow UI, you can:

* Compare model metrics across runs
* View confusion matrices under **Artifacts**
* Inspect model parameters and signatures
* Download the trained preprocessing pipeline
* Review `model_comparison.csv` for final rankings

---

## 📊 Models & Metrics

### Models Evaluated

* Logistic Regression
* Random Forest
* Decision Tree

### Metrics Tracked

* Accuracy
* Precision
* Recall
* **F1-Score** (used for final model selection)

---

## 📦 Requirements

* Python 3.8+
* pandas
* scikit-learn
* mlflow
* matplotlib
* joblib

---

## 📄 License

MIT
