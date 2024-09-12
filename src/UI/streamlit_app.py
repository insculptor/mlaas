####################################################################################
#####                    File: src/UI/streamlit_app.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                Streamlit Application UI Main File                        #####
####################################################################################

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="MLAAS", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# Add the Application root to Python path and set it as an environment variable
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_PATH)
os.environ["ROOT_PATH"] = ROOT_PATH  # Add ROOT_PATH to the environment
print(f"Added {ROOT_PATH} to the Python path and environment.")

from src.UI.htmltemplates import css
from src.UI.streamlit_pages import (
    data_upload_page,
    eda_viewer,
    model_training_page,
)

# Load environment variables
load_dotenv()

# Path to data directories from .env
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")

# Page configuration and styling
st.markdown(css, unsafe_allow_html=True)

def main():
    st.sidebar.title("MLAAS - Navigation")
    pages = ["MLAAS - Home", "Upload Data", "EDA Viewer", "Train Model"]
    choice = st.sidebar.radio("Select a Page", pages)

    if choice == "MLAAS - Home":
        st.write("Welcome to MLAAS! Please navigate to the appropriate page to upload data, perform EDA, or train models.")
    elif choice == "Upload Data":
        data_upload_page()
    elif choice == "EDA Viewer":
        eda_viewer()
    elif choice == "Train Model":
        model_training_page()

if __name__ == "__main__":
    main()
