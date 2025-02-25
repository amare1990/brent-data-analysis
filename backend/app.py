from flask import Flask, jsonify
import pandas as pd
import sys


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

if __name__ == "__main__":
    app.run(debug=True)
