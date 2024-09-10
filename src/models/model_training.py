####################################################################################
#####                     File: src/models/model_training.py                   #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                  Model Training using PyCaret and MLFlow                 #####
####################################################################################

import pandas as pd
import streamlit as st
from pycaret.classification import setup, compare_models, finalize_model, save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models
from pycaret.clustering import setup as cluster_setup, create_model
from src.models.mlflow_logging import log_experiment

def train_model(data, model_type="Base Model", task="classification"):
    """
    Trains a machine learning model on the given dataset using PyCaret.

    Parameters:
    data (pd.DataFrame): The dataset to train on.
    model_type (str): Either "Base Model" or "Best Model" for hyperparameter tuning.
    task (str): Type of ML task: "classification", "regression", or "clustering".

    Returns:
    None
    """
    try:
        # Define target column based on user input for supervised learning tasks
        if task in ["classification", "regression"]:
            target_column = st.selectbox("Select the target variable", data.columns)
        
        # Set up and train the model
        if task == "classification":
            exp = setup(data, target=target_column, silent=True)
            if model_type == "Base Model":
                model = compare_models(n_select=1)
            elif model_type == "Best Model":
                model = compare_models(n_select=1, turbo=False)  # Hyperparameter tuning
            final_model = finalize_model(model)
        elif task == "regression":
            exp = reg_setup(data, target=target_column, silent=True)
            if model_type == "Base Model":
                model = reg_compare_models(n_select=1)
            elif model_type == "Best Model":
                model = reg_compare_models(n_select=1, turbo=False)
            final_model = finalize_model(model)
        elif task == "clustering":
            exp = cluster_setup(data, silent=True)
            model = create_model('kmeans')  # Default to KMeans for clustering
            final_model = finalize_model(model)
        else:
            st.error("Unsupported task type.")
            return

        # Save the model
        model_name = f"{task}_{model_type.lower()}_model"
        save_model(final_model, model_name)

        # Log the experiment based on model type
        if model_type == "Base Model":
            log_experiment(model_name, model_type, task, run_type="Base Model")
        elif model_type == "Best Model":
            log_experiment(model_name, model_type, task, run_type="Fine-tuning")

        st.success(f"{model_type} training completed successfully for {task}. Model saved as {model_name}.")

    except Exception as e:
        st.error(f"Error during model training: {e}")
