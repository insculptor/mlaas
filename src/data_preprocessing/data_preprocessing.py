####################################################################################
#####               File: src/data_preprocessing/data_preprocessing.py         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                Data Preprocessing and EDA Utilities                      #####
####################################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def perform_eda(data):
    """
    Performs exploratory data analysis (EDA) on the given dataset and returns a set of plots.
    
    Parameters:
    data (pd.DataFrame): The dataset to perform EDA on.

    Returns:
    eda_results (dict): A dictionary containing the generated plots.
    """
    eda_results = {'plots': []}

    try:
        # Generate a pairplot for feature relationships (for numeric columns)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 1:
            st.subheader("Pairplot for Numeric Features")
            fig = sns.pairplot(data[numeric_columns])
            st.pyplot(fig)
            eda_results['plots'].append(fig)

        # Generate correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        eda_results['plots'].append(fig)

        # Generate histograms for all numeric columns
        for column in numeric_columns:
            st.subheader(f"Distribution of {column}")
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)
            eda_results['plots'].append(fig)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        return eda_results

    except Exception as e:
        st.error(f"Error during EDA: {e}")
        return eda_results
