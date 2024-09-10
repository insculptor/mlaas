####################################################################################
#####                     File: src/UI/streamlit_pages.py                      #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                    Streamlit App Web Pages Helper File                   #####
####################################################################################

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.utils.file_handling import handle_file_upload
from src.data_preprocessing.data_preprocessing import perform_eda
from src.models.model_training import train_model
from src.UI.htmltemplates import css

# Load environment variables
load_dotenv()

# Paths from environment
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")

# Custom CSS for styling
st.markdown(css, unsafe_allow_html=True)

## Page for uploading data
def data_upload_page():
    """Page for uploading data files."""
    st.markdown("<h1 style='text-align: center;'>📊 MLAAS - Upload Data</h1>", unsafe_allow_html=True)
    st.header("Upload CSV/XLS Dataset for Processing")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        # Handle the uploaded file
        data, file_path = handle_file_upload(uploaded_file, RAW_DATA_PATH)
        st.success(f"File uploaded successfully: {file_path}")
        st.write("Preview of the data:")
        st.dataframe(data.head())

        # Save to processed path
        processed_path = os.path.join(PROCESSED_DATA_PATH, os.path.basename(file_path))
        data.to_csv(processed_path, index=False)
        st.info(f"File saved to {processed_path}")

## Page for viewing EDA results
def eda_viewer():
    """Page for performing and viewing Exploratory Data Analysis (EDA)."""
    st.markdown("<h1 style='text-align: center;'>📊 MLAAS - EDA Viewer</h1>", unsafe_allow_html=True)
    st.header("Perform Exploratory Data Analysis")

    # Select the processed file to perform EDA on
    processed_files = os.listdir(PROCESSED_DATA_PATH)
    file_choice = st.selectbox("Select a processed file", processed_files)

    if file_choice:
        file_path = os.path.join(PROCESSED_DATA_PATH, file_choice)
        data = pd.read_csv(file_path)
        st.write("Selected data preview:")
        st.dataframe(data.head())

        # Perform EDA
        eda_results = perform_eda(data)
        st.write("EDA Results:")
        for plot in eda_results['plots']:
            st.pyplot(plot)

## Page for training models
def model_training_page():
    """Page for configuring and training machine learning models."""
    st.markdown("<h1 style='text-align: center;'>🤖 MLAAS - Train Models</h1>", unsafe_allow_html=True)
    st.header("Model Training and Tuning")

    # Select the processed file for model training
    processed_files = os.listdir(PROCESSED_DATA_PATH)
    file_choice = st.selectbox("Select a processed file", processed_files)

    if file_choice:
        file_path = os.path.join(PROCESSED_DATA_PATH, file_choice)
        data = pd.read_csv(file_path)
        st.write("Selected data preview:")
        st.dataframe(data.head())

        # Model training options
        model_type = st.selectbox("Select Model Type", ["Base Model", "Best Model"])

        if st.button("Train Model"):
            st.info(f"Training {model_type.lower()} on the selected dataset...")
            train_model(data, model_type)
            st.success(f"Model {model_type.lower()} trained successfully!")

