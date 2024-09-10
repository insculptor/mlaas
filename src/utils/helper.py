####################################################################################
#####                       File: src/utils/helper.py                          #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/10/2024                              #####
#####                 General Functions for Applicatation Dev                  #####
####################################################################################

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")

def list_files_and_dirs(directory, exclude_dirs=None):
    """
    List all files and directories recursively within a given directory,
    excluding specified directories.
    
    :param directory: Directory to scan
    :param exclude_dirs: List of directories to exclude
    :return: List of all files and directories
    """
    if exclude_dirs is None:
        exclude_dirs = []

    # Convert exclude_dirs to absolute paths
    exclude_dirs = [os.path.abspath(os.path.join(directory, d)) for d in exclude_dirs]

    result = []

    for root, dirs, files in os.walk(directory):
        # Convert root to absolute path
        root_abs = os.path.abspath(root)
        
        # Exclude directories that are in the exclude_dirs list
        dirs[:] = [d for d in dirs if os.path.join(root_abs, d) not in exclude_dirs]

        # Add directories and files to the result list
        for name in dirs:
            result.append(os.path.join(root, name))
        for name in files:
            result.append(os.path.join(root, name))

    return result

# Example usage
directory = ROOT_DIR
exclude_dirs = ['__pycache__', '.venv', '.git']
files_and_dirs = list_files_and_dirs(directory, exclude_dirs)

for item in files_and_dirs:
    print(item)
