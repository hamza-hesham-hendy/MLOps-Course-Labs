"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pickle
import os

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    os.makedirs("artifacts", exist_ok=True)
    transformer_path = "artifacts/column_transformer.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(col_transf, f)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(transformer_path, artifact_path="preprocess")

    return col_transf, X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))
    # Log model
    mlflow.sklearn.log_model(log_reg, "logistig_regression", signature=signature,input_example=X_train.head(5))
    ### Log the data
    dataset = mlflow.data.from_pandas(X_train,name="train_data")
    mlflow.log_input(dataset, context="training")

    return log_reg

def train_random_forest(X_train, y_train):
    """
    Train a random forest model.
    """
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, rf.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest", signature=signature, input_example=X_train.head(5))
    ### Log the data
    dataset = mlflow.data.from_pandas(X_train,name="train_data")
    mlflow.log_input(dataset, context="training")

    return rf


def train_decision_tree(X_train, y_train):
    """
    Train a decision tree model.
    """
    dt = DecisionTreeClassifier(random_state=1912)
    dt.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, dt.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(dt, "decision_tree", signature=signature, input_example=X_train.head(5))

    ### Log the data
    dataset = mlflow.data.from_pandas(X_train,name="train_data")
    mlflow.log_input(dataset, context="training")
    return dt

def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Bank_Churn_Prediction_multimodel")

    df = pd.read_csv("dataset/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)
    
    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="multimodel_comparison"):
        results = []

        models = [
            ("logistic_regression", train_logistic_regression, {"version": "v1.0.0", "max_iter": 1000}),
            ("random_forest", train_random_forest, {"version": "v1.0.0", "n_estimators": 300, "random_state": 42}),
            ("decision_tree", train_decision_tree, {"version": "v1.0.0", "random_state": 42}),
        ]

        for model_name, train_fn, params in models:
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log params for this model
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                # Train (training function logs model + input sample)
                model = train_fn(X_train, y_train)
                # Predict + metrics
                y_pred = model.predict(X_test)
                
                ### Log metrics after calculating them
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
                mlflow.log_metric("precision", precision_score(y_test, y_pred))
                mlflow.log_metric("recall", recall_score(y_test, y_pred))
                mlflow.log_metric("f1", f1_score(y_test, y_pred))

                ### Log tag
                mlflow.set_tag("version", params["version"])
                mlflow.set_tag("model", model_name)
                
                # Confusion matrix plot
                conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
                conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_).plot()
                conf_mat_disp.plot()
                
                # Log the image as an artifact in MLflow
                os.makedirs("artifacts", exist_ok=True)
                plot_path = f"artifacts/confusion_matrix_{model_name}.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)
                # plt.show()

                # Collect results for final comparison
                results.append({
                    "model": model_name,
                    "version": params["version"],
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                })
                # ---- Final multi-model evaluation artifact (comparison table) ----
        results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
        results_df.to_csv("model_comparison.csv", index=False)
        mlflow.log_artifact("model_comparison.csv", artifact_path="comparison")


if __name__ == "__main__":
    main()
