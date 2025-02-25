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

from scripts.lstm import LSTMTimeSeries

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


@app.route("/api/describe", methods=["GET"])
def describe_data():
    """Statistical description of data"""
    # return jsonify({"statistical summary": str(workflow.data.describe())})
    return jsonify(workflow.data.describe().to_dict())



@app.route("/api/arima", methods=["GET"])
def get_arima_results():
    """API endpoint to fetch ARIMA results"""
    return jsonify({"arima_summary": str(arima_summary)})


@app.route('/api/compare_models', methods=['GET'])
def compare_models():
    """Compare model performance using RMSE, MAE, and R² Score."""

    print(f"\n{'*'*100}\n")
    print("Comparing Model Performance\n")

    # --------------------- ARIMA Model Evaluation ---------------------
    arima_preds = statistical_model.arima_result.forecast(steps=len(statistical_model.test))
    arima_rmse = np.sqrt(
        mean_squared_error(
            statistical_model.test['Price'],
            arima_preds))
    arima_mae = mean_absolute_error(statistical_model.test['Price'], arima_preds)
    arima_r2 = r2_score(statistical_model.test['Price'], arima_preds)

    # --------------------- Bayesian Inference Evaluation -----------------
    bayes_preds = [statistical_model.train['Price'].mean()] * len(statistical_model.test)
    bayesian_rmse = np.sqrt(
        mean_squared_error(
            statistical_model.test['Price'],
            bayes_preds))
    bayesian_mae = mean_absolute_error(statistical_model.test['Price'], bayes_preds)
    bayesian_r2 = r2_score(statistical_model.test['Price'], bayes_preds)

    # --------------------- GARCH Model Evaluation ---------------------
    # Squared returns
    actual_volatility = statistical_model.test['Price'].pct_change() ** 2
    garch_forecast = statistical_model.garch_result.forecast(
        start=len(statistical_model.train) - len(statistical_model.test),
        horizon=len(statistical_model.test)
    )
    # Predicted volatility
    garch_preds = garch_forecast.variance.values[-1, :]

    min_len = min(len(actual_volatility.dropna()), len(garch_preds))
    actual_volatility = actual_volatility.dropna()[:min_len]
    garch_preds = garch_preds[:min_len]

    garch_rmse = np.sqrt(
        mean_squared_error(
            actual_volatility,
            garch_preds))
    garch_mae = mean_absolute_error(actual_volatility, garch_preds)
    garch_r2 = r2_score(actual_volatility, garch_preds)

    # --------------------- LSTM Model Evaluation ---------------------
    lstm_model = LSTMTimeSeries(data=statistical_model.data, epochs=10, batch_size=32)

    lstm_model, perf_metrics = lstm_model.fit()
    lstm_rmse = perf_metrics['rmse']
    lstm_mae = perf_metrics['mae']
    lstm_r2 = perf_metrics['r2']
    print(f"LSTM Performance Metrics:")
    print(f"RMSE: {perf_metrics['rmse']}")
    print(f"MAE: {perf_metrics['mae']}")
    print(f"R² Score: {perf_metrics['r2']}")

    # --------------------- Print Model Comparison ---------------------
    metrics = pd.DataFrame({
        "Model": ["ARIMA", "GARCH", "Bayesian Inference", "LSTM"],
        "RMSE": [arima_rmse, garch_rmse, bayesian_rmse, lstm_rmse],
        "MAE": [arima_mae, garch_mae, bayesian_mae, lstm_mae],
        "R² Score": [arima_r2, garch_r2, bayesian_r2, lstm_r2]
    })

    # Print the API response to debug
    print(metrics.to_string(index=False))

    response_data = metrics.to_dict(orient="records")
    print("Sending JSON response:", response_data)



    print(f"\n{'-'*100}\n")
    print("✅ Model with lowest RMSE and MAE is generally the best.")
    print("✅ Higher R² Score (closer to 1) indicates better model fit.\n")

    # Return the metrics as JSON
    return jsonify(response_data)

if __name__ =="__main__":
    app.run(debug=True)

