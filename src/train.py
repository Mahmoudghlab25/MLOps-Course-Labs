"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
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
from mlflow.models import infer_signature
import mlflow.sklearn
import logging_helper

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

    # # Log the transformer as an artifact
    # logging_helper.log_transformer(col_transf, "preprocessor")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
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
    signature = mlflow.models.infer_signature(X_train, log_reg.predict(X_train))
    logging_helper.log_transformer(log_reg, "logistic_regression_model", signature=signature)

    # Log model
    logging_helper.log_transformer(log_reg, "logistic_regression_model")
    ### Log the data
    logging_helper.log_artifact("data/Churn_Modelling.csv", "churn_modelling_data.csv")
    return log_reg


def train_model(X_train, y_train, model_type="logistic_regression"):
    """
    Train a logistic regression model or random forest model
    or decision tree model based on the input model type.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
        RandomForestClassifier: trained random forest classifier model
        DecisionTreeClassifier: trained decision tree classifier model
    """
    m_obj = None
    if model_type == "logistic_regression":
        m_obj = LogisticRegression(max_iter=1000)
    
    elif model_type == "random_forest":
        m_obj = RandomForestClassifier(n_estimators=100, random_state=1912)
    elif model_type == "decision_tree":
        m_obj = DecisionTreeClassifier(random_state=1912, criterion="gini", max_depth=None)

    m_obj.fit(X_train, y_train)
    

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, m_obj.predict(X_train))
    logging_helper.log_transformer(m_obj, f"{model_type}_model", signature=signature)

    # Log model
    logging_helper.log_transformer(m_obj, f"{model_type}_model")

    ### Log the data
    logging_helper.log_artifact("data/Churn_Modelling.csv", "churn_modelling_data.csv")
    return m_obj

def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Bank Consumer Churn Prediction")

    ### Start a new run and leave all the main function code as part of the experiment
    r_name = "Decision Tree Model Training"
    curr_run = mlflow.start_run(run_name=r_name)

    df = pd.read_csv("data/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter
    type = "decision_tree"
    if type == "logistic_regression":
        logging_helper.log_parameters({"max_iter": 1000})
    elif type == "random_forest":
        logging_helper.log_parameters({"n_estimators": 100})
    elif type == "Decision Tree":
        logging_helper.log_parameters({"criterion": "gini", "max_depth": None})


    # model = train(X_train, y_train)
    
    model = train_model(X_train, y_train, model_type=type)
    
    y_pred = model.predict(X_test)

    ### Log metrics after calculating them
    logging_helper.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        })


    ### Log tag
    logging_helper.log_tags({"model_type": type, "author": "Mahmoud"})
    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    plt.savefig("confusion_matrix.png")
    # Log the image as an artifact in MLflow
    logging_helper.log_artifact("confusion_matrix.png", "confusion_matrix.png")
    
    plt.show()
    # End the MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
