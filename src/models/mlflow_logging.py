####################################################################################
#####                     File: src/models/mlflow_logging.py                   #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                       MLFlow Logging Utilities                           #####
####################################################################################

import os

import mlflow
import mlflow.sklearn
import streamlit as st
from dotenv import load_dotenv
from mlflow import log_metric, log_param

# Load environment variables
load_dotenv()

# Set the MLFlow tracking URI from the .env file
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Enable autologging for sklearn
mlflow.sklearn.autolog()

def log_base_model_experiment(model_name, model, model_type, task, metrics=None, X=None):
    """
    Logs the base model experiment to MLFlow.
    Parameters:
    model_name (str): The name of the model.
    model: The trained model to be logged.
    model_type (str): The type of model (Base Model or Best Model).
    task (str): Type of ML task: "classification", "regression".
    metrics (dict): Dictionary containing performance metrics (accuracy, MSE, etc.).
    X: Input data for model signature logging.
    Returns:
    None
    """
    try:
        with mlflow.start_run(run_name=f"{model_name}_{model_type}"):
            log_param("model_name", model_name)
            log_param("model_type", model_type)
            log_param("task", task)

            # Log metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (float, int)):
                        log_metric(metric_name, metric_value)

            # Log model with input signature
            if X is not None:
                mlflow.sklearn.log_model(model, model_name, input_example=X)
            else:
                mlflow.sklearn.log_model(model, model_name)

            st.success(f"Base model experiment logged successfully for {model_name} in MLFlow.")
    
    except Exception as e:
        st.error(f"Error logging base model experiment to MLFlow: {e}")


def log_best_model_experiment(model_name, model, model_type, task, metrics=None, X=None):
    """
    Logs the best model experiment with parent and child runs.
    Parameters:
    model_name (str): The name of the best model.
    model: The best model to be logged.
    model_type (str): The type of model (Base Model or Best Model).
    task (str): Type of ML task: "classification", "regression".
    metrics (dict): Dictionary containing models and performance metrics.
    X: Input data for model signature logging.
    Returns:
    None
    """
    try:
        with mlflow.start_run(run_name=f"{model_name}_Best Model", nested=False) as parent_run:
            parent_run_id = parent_run.info.run_id
            st.info(f"Started parent run with ID: {parent_run_id}")

            # Log best model details in the parent run
            log_param("best_model", model_name)
            mlflow.sklearn.log_model(model, model_name, input_example=X)

            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (float, int)):
                        log_metric(metric_name, metric_value)

            # Log child models
            for i, (name, child_model) in enumerate(metrics["models"].items(), 1):
                with mlflow.start_run(run_name=f"{name}_Child_{i}", nested=True) as child_run:
                    log_param("model_name", name)
                    log_param("model_type", model_type)
                    log_param("task", task)

                    # Log metrics for child model
                    for metric_name, metric_value in metrics["scores"][name].items():
                        if isinstance(metric_value, (float, int)):
                            log_metric(metric_name, metric_value)

                    # Log child model with input signature
                    mlflow.sklearn.log_model(child_model, name, input_example=X)

            st.success(f"Fine-tuning completed for {model_name}. Parent and child runs logged in MLFlow.")

    except Exception as e:
        st.error(f"Error logging best model experiment to MLFlow: {e}")
