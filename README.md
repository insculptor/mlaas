# MLAAS - Machine Learning as a Service

MLAAS is a platform that allows users to upload datasets and automatically apply machine learning models for both supervised and unsupervised learning tasks. The application provides a user-friendly interface built with Streamlit, enabling users to perform exploratory data analysis (EDA), train models using PyCaret, and track experiments with MLFlow. Users can download trained models and artifacts, making it easy to integrate the results into their own projects.

## Architecture

The architecture of the MLAAS application is outlined in the diagram below:

![MLAAS Application Flow](images/MLAAS_Application_Flow.png)

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Environment Variables](#environment-variables)
6. [How it Works](#how-it-works)
7. [Acknowledgements](#acknowledgements)

## Features

- **Streamlit UI**: A user-friendly web interface for uploading data files and configuring machine learning tasks.
- **Dual Learning Modes**: Users can select both supervised and unsupervised learning tasks simultaneously, with checkboxes enabled by default.
- **EDA & Visualization**: Perform exploratory data analysis and generate insightful plots, all logged in MLFlow for easy access and download.
- **Model Training**: Utilize PyCaret to train both base models and fine-tuned models for the best performance.
- **MLFlow Integration**: Comprehensive experiment tracking, including models, metrics, and EDA artifacts.
- **Model & Artifact Download**: Option to download the trained models and all associated artifacts as a zip file directly from the UI.

## Project Structure

```
mlaas/
│
├── .streamlit/                        # Streamlit configuration files
│   └── config.toml                    # Streamlit app configuration
│
├── data/                              # Data directory
│   ├── processed/                     # Processed datasets
│   └── raw/                           # Raw datasets
│
├── models/                            # Directory to store trained models
│
├── src/                               # Source code directory
│   ├── data_preprocessing/            # Data processing and EDA scripts
│   │   └── data_preprocessing.py      # Handles data cleaning, preprocessing, and EDA
│   │
│   ├── models/                        # Model-related scripts
│   │   ├── mlflow_logging.py          # MLFlow logging utilities
│   │   └── model_training.py          # Handles model training using PyCaret
│   │
│   ├── UI/                            # User Interface components
│   │   ├── htmltemplates.py           # HTML templates and CSS for UI styling
│   │   ├── streamlit_app.py           # Main Streamlit app file
│   │   └── streamlit_pages.py         # Streamlit page logic
│   │
│   └── utils/                         # Utility scripts
│       ├── file_handling.py           # File upload handling and validation
│       └── helper.py                  # Helper functions
│
├── .env                               # Environment variables configuration
├── .gitignore                         # Git ignore file
├── config.yaml                        # Configuration file for model parameters
├── LICENSE                            # License file
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- [PyCaret](https://pycaret.org/)
- [MLFlow](https://mlflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mlaas.git
   cd mlaas
   ```

2. **Create a Python virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables in `.env` (see [Environment Variables](#environment-variables)):**

   ```bash
   MLFLOW_TRACKING_URI=http://localhost:5000
   ```

5. **Run the Streamlit application:**

   ```bash
   streamlit run src/UI/streamlit_app.py
   ```

## Usage

### 1. **Upload Dataset**

   - Navigate to the home page of the application.
   - Upload a CSV or XLS file containing your dataset.
   - The application will display a preview of the data.

### 2. **Select Learning Tasks**

   - **Supervised Learning**: For tasks like regression or classification.
   - **Unsupervised Learning**: For tasks like clustering or dimensionality reduction.
   - Both checkboxes are selected by default, allowing you to perform both tasks simultaneously.

### 3. **Configure Model Preferences**

   - Choose between training a **Base Model** (quick fit) or the **Best Model** (fine-tuned with hyperparameter tuning).

### 4. **Perform Exploratory Data Analysis (EDA)**

   - The application will automatically perform EDA.
   - View generated plots like histograms, box plots, scatter plots, and correlation heatmaps.
   - All EDA artifacts are logged in MLFlow for later viewing and downloading.

### 5. **Train Models**

   - Click on the "Train Models" button.
   - The system will train models for the selected learning tasks using PyCaret.
   - Progress bars and logs will show the training status.

### 6. **View Results and Download Models**

   - After training, view model performance metrics directly in the UI.
   - Option to download the trained models or a zip file containing models and artifacts.

### 7. **Experiment Tracking with MLFlow**

   - Access the MLFlow UI to see detailed experiment logs.
   - Compare different models, view metrics, and download artifacts.

## Environment Variables

The application uses a `.env` file for configuration. Ensure you set the following variables:

```bash
# MLFlow Tracking URI
MLFLOW_TRACKING_URI=http://localhost:5000

# (Optional) Hugging Face Token if using Hugging Face models
HUGGINGFACE_TOKEN=your_huggingface_token
```

- **MLFLOW_TRACKING_URI**: The URI where your MLFlow server is running.
- **HUGGINGFACE_TOKEN**: Token for accessing private models on Hugging Face (if applicable).

## How it Works

### 1. **Streamlit UI**

   - Provides an interface for file uploads and task configuration.
   - Displays data previews, EDA results, and model performance.

### 2. **Data Processing and EDA**

   - Uploaded data is processed in `src/data_preprocessing/data_preprocessing.py`.
   - Columns are renamed to `C1, C2, ..., CN` to anonymize data.
   - EDA is performed, generating plots and statistics.

### 3. **Model Training with PyCaret**

   - Models are trained using PyCaret in `src/models/model_training.py`.
   - Supports both supervised and unsupervised learning.
   - Options for quick training (base model) or fine-tuning (best model).

### 4. **MLFlow Experiment Tracking**

   - All experiments, models, and artifacts are logged using MLFlow.
   - EDA plots and model metrics are stored as artifacts.
   - Parent-child experiment structure allows for organized tracking.

### 5. **Model and Artifact Download**

   - Users can download the trained models directly from the UI.
   - Option to download all artifacts, including EDA plots and logs, as a zip file.

## Acknowledgements

- **[PyCaret](https://pycaret.org/)**: For simplifying the machine learning model training process.
- **[Streamlit](https://streamlit.io/)**: For providing an easy-to-use interface for the application.
- **[MLFlow](https://mlflow.org/)**: For experiment tracking and model management.
- **[Pandas](https://pandas.pydata.org/)** and **[NumPy](https://numpy.org/)**: For data manipulation and numerical computations.