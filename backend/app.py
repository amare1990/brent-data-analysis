from flask import Flask, jsonify
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"
# Add the base directory to sys.path to allow imports from scripts/
sys.path.append(BASE_DIR)

from scripts.data_analysis import BrentDataAnalysis
from scripts.statistical_modeling import StatisticalModel

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests


workflow = BrentDataAnalysis("../data/BrentOilPrices.csv")
workflow.overview_of_data()
workflow.describe_data()
workflow.identify_missing_values()
workflow.remove_duplicates()
workflow.sort_data()
workflow.plot_time_series()

df = pd.read_csv("../data/processed_data.csv")
statistical_model = StatisticalModel(df)

arima_summary = statistical_model.fit_arima()



@app.route("/api/data", methods=["GET"])
def get_data():
    """API endpoint to fetch oil price data"""
    return workflow.data.to_json()


@app.route("api/describe", methods=["GET"])
def describe_data():
    """Statistical description of data"""
    return workflow.data.describe().to_json()


@app.route("/api/arima", methods=["GET"])
def get_arima_results():
    """API endpoint to fetch ARIMA results"""
    return jsonify({"arima_summary": str(arima_summary)})


@app.route('/api/compare_models', methods=['GET'])
def compare_models():
    """Compare model performance using RMSE, MAE, and R² Score."""

    # Dummy test data and model results for the sake of example
    # Replace this part with your actual models' predictions
    test = pd.DataFrame({'Price': np.random.randn(100)})
    train = pd.DataFrame({'Price': np.random.randn(100)})

    # ARIMA Model
    arima_preds = np.random.randn(len(test))
    arima_rmse = np.sqrt(mean_squared_error(test['Price'], arima_preds))
    arima_mae = mean_absolute_error(test['Price'], arima_preds)
    arima_r2 = r2_score(test['Price'], arima_preds)

    # Bayesian Inference Model
    bayes_preds = [train['Price'].mean()] * len(test)
    bayesian_rmse = np.sqrt(mean_squared_error(test['Price'], bayes_preds))
    bayesian_mae = mean_absolute_error(test['Price'], bayes_preds)
    bayesian_r2 = r2_score(test['Price'], bayes_preds)

    # GARCH Model
    garch_preds = np.random.randn(len(test))
    garch_rmse = np.sqrt(mean_squared_error(test['Price'], garch_preds))
    garch_mae = mean_absolute_error(test['Price'], garch_preds)
    garch_r2 = r2_score(test['Price'], garch_preds)

    # LSTM Model
    lstm_rmse = np.random.rand()
    lstm_mae = np.random.rand()
    lstm_r2 = np.random.rand()

    # Prepare the metrics data for frontend
    metrics = [
        {"model": "ARIMA", "rmse": arima_rmse, "mae": arima_mae, "r2": arima_r2},
        {"model": "Bayesian Inference", "rmse": bayesian_rmse, "mae": bayesian_mae, "r2": bayesian_r2},
        {"model": "GARCH", "rmse": garch_rmse, "mae": garch_mae, "r2": garch_r2},
        {"model": "LSTM", "rmse": lstm_rmse, "mae": lstm_mae, "r2": lstm_r2},
    ]

    return jsonify(metrics)

if __name__ == "__main__":
    app.run(debug=True)

