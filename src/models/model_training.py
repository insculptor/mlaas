####################################################################################
#####                     File: src/models/model_training.py                   #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                  Model Training using sklearn and MLFlow                 #####
####################################################################################

import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.models.mlflow_logging import (
    log_base_model_experiment,
    log_best_model_experiment,
)


def train_model(data, model_type="Base Model", task="classification"):
    """
    Trains a machine learning model on the given dataset without PyCaret.

    Parameters:
    data (pd.DataFrame): The dataset to train on.
    model_type (str): Either "Base Model" or "Best Model" for hyperparameter tuning.
    task (str): Type of ML task: "classification" or "regression".

    Returns:
    None
    """
    try:
        # Step 1: Define the target column based on user input for supervised learning tasks
        target_column = st.selectbox("Select the target variable", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Step 2: Identify categorical columns for OneHotEncoding
        categorical_columns = X.select_dtypes(include=['object']).columns

        # Step 3: Preprocess the categorical features using ColumnTransformer
        column_transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
            remainder='passthrough'
        )

        # Step 4: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.write(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

        # Step 5: Apply OneHotEncoding and scaling
        X_train = column_transformer.fit_transform(X_train)
        X_test = column_transformer.transform(X_test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 6: Encode target variable if it's categorical
        if y.dtypes == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Base Model
        if model_type == "Base Model":
            # Choose base model based on task
            if task == "classification":
                model = DecisionTreeClassifier()
            elif task == "regression":
                model = DecisionTreeRegressor()
            else:
                st.error("Unsupported task type.")
                return

            # Train and evaluate the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log results
            if task == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Base Model Accuracy: {accuracy}")
                metrics = {"accuracy": accuracy}
                log_base_model_experiment("DecisionTree_Base", model, model_type, task, metrics=metrics, X=X_train[:5])
            elif task == "regression":
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Base Model Mean Squared Error: {mse}")
                metrics = {"mse": mse}
                log_base_model_experiment("DecisionTree_Base", model, model_type, task, metrics=metrics, X=X_train[:5])

        # Best Model (train on multiple algorithms)
        elif model_type == "Best Model":
            models = []
            model_dict = {}
            scores_dict = {}

            # Choose algorithms based on task
            if task == "classification":
                models = [
                    ('RandomForest', RandomForestClassifier()),
                    ('LogisticRegression', LogisticRegression()),
                    ('BaggingClassifier', BaggingClassifier()),
                    ('AdaBoostClassifier', AdaBoostClassifier()),
                    ('GradientBoostingClassifier', GradientBoostingClassifier())
                ]
            elif task == "regression":
                models = [
                    ('LinearRegression', LinearRegression()),
                    ('SVR', SVR()),
                    ('BaggingRegressor', BaggingRegressor()),
                    ('AdaBoostRegressor', AdaBoostRegressor()),
                    ('GradientBoostingRegressor', GradientBoostingRegressor())
                ]
            else:
                st.error("Unsupported task type.")
                return

            # Train and evaluate models to find the best
            best_model = None
            best_score = float('-inf') if task == "classification" else float('inf')

            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"{name} Accuracy: {accuracy}")
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                    model_dict[name] = model
                    scores_dict[name] = {"accuracy": accuracy}
                elif task == "regression":
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"{name} Mean Squared Error: {mse}")
                    if mse < best_score:
                        best_score = mse
                        best_model = model
                    model_dict[name] = model
                    scores_dict[name] = {"mse": mse}

            # Log best model with parent-child logging
            metrics = {"models": model_dict, "scores": scores_dict}
            log_best_model_experiment(best_model.__class__.__name__, best_model, model_type, task, metrics=metrics, X=X_train[:5])

    except Exception as e:
        st.error(f"Error during model training: {e}")
