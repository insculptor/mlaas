####################################################################################
#####                     File: src/models/mlflow_logging.py                   #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                       MLFlow Logging Utilities                           #####
####################################################################################

import os
import mlflow
import streamlit as st
from mlflow import log_artifact, log_param, log_metric
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the MLFlow tracking URI from the .env file
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Enable autologging for sklearn
mlflow.sklearn.autolog()

def log_experiment(model_name, model_type, task, run_type="Base Model"):
    """
    Logs the model, parameters, and metrics to MLFlow. Handles both single runs for base models 
    and parent-child runs for fine-tuned models.

    Parameters:
    model_name (str): The name of the model.
    model_type (str): The type of model (Base Model or Best Model).
    task (str): Type of ML task: "classification", "regression", or "clustering".
    run_type (str): Type of run ("Base Model" or "Fine-tuning").

    Returns:
    None
    """
    try:
        if run_type == "Base Model":
            # Start a new experiment for base models
            with mlflow.start_run(run_name=f"{model_name}_{model_type}"):
                # Log parameters
                log_param("model_name", model_name)
                log_param("model_type", model_type)
                log_param("task", task)

                # Log model artifact
                log_artifact(f"{model_name}.pkl")

                # Example: Log a sample metric (replace with actual metrics)
                log_metric("accuracy", 0.95)

            st.success(f"Base model experiment logged successfully for {model_name} in MLFlow.")

        elif run_type == "Fine-tuning":
            # Start a parent run for fine-tuning
            with mlflow.start_run(run_name=f"{model_name}_Best Model", nested=False) as parent_run:
                parent_run_id = parent_run.info.run_id
                st.info(f"Started parent run with ID: {parent_run_id}")

                # Simulate multiple child runs (this can be part of hyperparameter tuning)
                for i in range(1, 4):
                    with mlflow.start_run(run_name=f"{model_name}_Fine-tuning_Trial_{i}", nested=True) as child_run:
                        # Log parameters for each fine-tuning run
                        log_param("trial_number", i)
                        log_param("model_name", model_name)
                        log_param("model_type", model_type)
                        log_param("task", task)

                        # Log metrics (replace with actual metrics)
                        log_metric("accuracy", 0.90 + i * 0.01)  # Example: dynamic metrics

                        # Log model artifact
                        log_artifact(f"{model_name}_trial_{i}.pkl")

                st.success(f"Fine-tuning completed for {model_name}. Parent run and child runs logged in MLFlow.")

    except Exception as e:
        st.error(f"Error logging experiment to MLFlow: {e}")