````md
# Bank Customer Churn Prediction (MLOps)

This repository contains a modular Machine Learning pipeline designed to predict bank customer churn. It leverages **MLflow** for experiment tracking, model management, artifact logging, and performance comparison across multiple classifiers.

---

## 📂 Project Structure

```text
MLOps-Course-Labs/
├── artifacts/                # Local storage for plots and pickled transformers
│   ├── column_transformer.pkl
│   └── confusion_matrix_*.png
├── dataset/
│   └── Churn_Modelling.csv   # Raw customer dataset
├── src/
│   ├── train.py              # Baseline single-model training (Logistic Regression)
│   └── multimodel_train.py   # Multi-model experiment with nested MLflow runs
├── mlruns/                   # MLflow local tracking directory
├── model_comparison.csv      # Auto-generated CSV comparing all model metrics
├── requirements.txt          # Python dependencies
└── README.md
````

---

## 🚀 Features

* **Automated Preprocessing**

  * Handles class imbalance via downsampling
  * Uses `ColumnTransformer` for scaling numerical features and one-hot encoding categorical features

* **Experiment Tracking**

  * Logs parameters, accuracy, precision, recall, and F1-score using MLflow

* **Model Signatures**

  * Stores input/output schemas in MLflow for reproducibility and deployment readiness

* **Nested Runs**

  * `multimodel_train.py` organizes multiple models under a single parent experiment

* **Automated Model Comparison**

  * Generates `model_comparison.csv` ranked by F1-score

---

## 🛠️ Installation

Navigate to the project directory and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Usage

### 1. Start MLflow Tracking Server

In a separate terminal:

```bash
mlflow ui --port 5000
```

### 2. Run Experiments

Train the baseline Logistic Regression model:

```bash
python src/train.py
```

Run the full multi-model comparison (Logistic Regression, Random Forest, Decision Tree):

```bash
python src/multimodel_train.py
```

### 3. Analyze Results

Open the MLflow UI in your browser:

```
http://localhost:5000
```

From the UI, you can:

* Compare metrics across models
* Inspect confusion matrix plots under **Artifacts**
* Download the `column_transformer.pkl` for inference
* Review `model_comparison.csv` for final rankings

---

## 📊 Models & Metrics

**Models Included**

* Logistic Regression
* Random Forest
* Decision Tree

**Metrics Tracked**

* Accuracy
* Precision
* Recall
* F1-Score (primary metric for model ranking)

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

```
```
=======
