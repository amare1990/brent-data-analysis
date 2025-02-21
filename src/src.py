"""Module for automatic running of end-to-end."""

import os
import sys


# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"


from scripts.data_analysis import BrentDataAnalysis


if __name__ == "__main__":

  file_path = f"{base_dir}/data/BrentOilPrices.csv"
  save_path = f"{base_dir}/data/processed_data.csv"

  workflow = BrentDataAnalysis(file_path)
  workflow.overview_of_data()
  workflow.describe_data()
  workflow.identify_missing_values()
  workflow.remove_duplicates()
  workflow.save_processed_data(save_path)
  workflow.plot_time_series()
