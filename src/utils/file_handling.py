####################################################################################
#####                     File: src/utils/file_handling.py                     #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                   File Upload and Validation Utilities                   #####
####################################################################################

import os
import pandas as pd
import streamlit as st

def handle_file_upload(uploaded_file, save_path):
    """
    Handles file upload, processes it into a Pandas DataFrame, and saves it.

    Parameters:
    uploaded_file: File uploaded by the user (Streamlit file_uploader object)
    save_path: Path where the uploaded file will be saved

    Returns:
    data (pd.DataFrame): The dataframe created from the uploaded file
    file_path (str): The full path to the saved file
    """
    try:
        # Determine the file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Read the uploaded file into a Pandas DataFrame
        if file_extension == '.csv':
            data = pd.read_csv(uploaded_file)
        elif file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None

        # Create the save path if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the uploaded file to the designated path
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded and saved successfully: {file_path}")
        return data, file_path

    except Exception as e:
        st.error(f"Error while processing the uploaded file: {e}")
        return None, None
