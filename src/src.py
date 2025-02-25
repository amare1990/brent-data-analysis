"""Module for automatic running of end-to-end."""

import os
import sys

import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

from scripts.change_point_analysis import ChangePointAnalysis
from scripts.data_analysis import BrentDataAnalysis
from scripts.statistical_modeling import StatisticalModel


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"


if __name__ == "__main__":

    # Run the pipeline processes for the data analysis module
    file_path = f"{base_dir}/data/BrentOilPrices.csv"
    save_path = f"{base_dir}/data/processed_data.csv"

    workflow = BrentDataAnalysis(file_path)
    workflow.overview_of_data()
    workflow.describe_data()
    workflow.identify_missing_values()
    workflow.remove_duplicates()
    workflow.sort_data()
    workflow.plot_time_series()
    workflow.save_processed_data(save_path)

    # Run the pipeline processes for the statistical data modeling modules
    processed_data_path = f"{base_dir}/data/processed_data.csv"
    df_processed = pd.read_csv(processed_data_path)
    statistical_model = StatisticalModel(df_processed)
    result = statistical_model.check_stationarity()
    adf_statistic = result['ADF Statistic']
    p_value = result['p-value']
    print(f"ADF Statistic: {adf_statistic}, \np-value: {p_value}")

    statistical_model.plot_acf_pacf()

    statistical_model.fit_arima()

    statistical_model.fit_garch()

    trace = statistical_model.bayesian_inference()

    # var_result = statistical_model.fit_var()
    # markov_switching_result = statistical_model.fit_markov_switching_arima()

    lstm_model, perf_metrics = statistical_model.fit_lstm(
        epochs=10, batch_size=32)

    statistical_model.compare_models()

    # Run the pipeline for change point analysis
    print(f"\n{'*'*100}\n")
    processed_data = f"{base_dir}/data/processed_data.csv"
    change_point_analysis = ChangePointAnalysis(
        processed_data, "1987-05-20", "2022-09-30")
    change_point_analysis.download_data()
    change_point_analysis.merge_data()
    change_point_analysis.scale_data()
    change_point_analysis.compare_models()
