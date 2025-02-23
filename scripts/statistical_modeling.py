"""Statistical modeling."""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc as pm
import arviz as az

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"


class StatisticalModel:
    def __init__(self, data):
        self.data = data
        self.train, self.test = self.split_data()

    def split_data(self, train_ratio=0.8):
        """Split time series data into training and testing sets."""
        train_size = int(len(self.data) * train_ratio)
        return self.data.iloc[:train_size], self.data.iloc[train_size:]

    def check_stationarity(self):
        print(f"\n{'*'*70}\n")
        print("Checking for stationarity using ADF test.")
        result = adfuller(self.data['Price'])
        return {'ADF Statistic': result[0], 'p-value': result[1]}

    def plot_acf_pacf(self, lags=30):
        """Plot ACF and PACF to determine AR and MA orders."""
        print(f"\n{'*'*70}\n")
        print("Plotting ACF and PACF to determine AR and MA orders.")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Plot ACF (AutoCorrelation Function)
        plot_acf(self.data['Price'], lags=lags, ax=ax[0])
        ax[0].set_title("Autocorrelation Function (ACF)")
        ax[0].axhline(y=0, linestyle='--', color='gray')

        # Plot PACF (Partial AutoCorrelation Function)
        plot_pacf(self.data['Price'], lags=lags, ax=ax[1])
        ax[1].set_title("Partial Autocorrelation Function (PACF)")

        plt.tight_layout()
        plt.savefig(
            f'{base_dir}/notebooks/plots/acf_pacf_plot.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()

    def fit_arima(self, order=(1, 1, 1)):
        """Fit an ARIMA model"""
        print(f"\n{'*'*100}\n")
        print("Fitting ARIMA model.\n")
        model = ARIMA(self.train['Price'], order=order)
        self.arima_result = model.fit()

        # Forecast
        arima_preds = self.arima_result.forecast(steps=len(self.test))
        # Compute Metrics
        arima_rmse = np.sqrt(mean_squared_error(self.test['Price'], arima_preds))
        arima_mae = mean_absolute_error(self.test['Price'], arima_preds)
        arima_r2 = r2_score(self.test['Price'], arima_preds)

        print(f"📊 ARIMA Performance:")
        print(f"✅ RMSE: {arima_rmse}")
        print(f"✅ MAE: {arima_mae}")
        print(f"✅ R² Score: {arima_r2}")

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index, self.test['Price'], label="Actual Price")
        plt.plot(self.test.index, arima_preds, linestyle="dashed", label="Predicted Price")
        plt.legend()
        plt.title("ARIMA Predictions vs Actual")
        plt.savefig(
            f"{base_dir}/notebooks/plots/arima_prediction.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.show()
        print(self.arima_result.summary())

    def fit_garch(self):
        """Fit a GARCH model"""
        print(f"\n{'*'*100}\n")
        print("Fitting GARCH model.\n")
        garch = arch_model(self.train['Price'].pct_change().dropna(), vol='Garch', p=1, q=1)
        self.garch_result = garch.fit()

        # Forecast volatility
        # Adjust start and horizon to ensure they are within the data range
        horizon = min(len(self.test), len(self.train))  # Limit horizon to training data size
        start = len(self.train) - horizon
        garch_forecast = self.garch_result.forecast(start=start, horizon=horizon)

        volatility_preds = garch_forecast.variance.values[-1, :]
        garch_preds = volatility_preds

        # Actual squared returns
        actual_volatility = self.test['Price'].pct_change() ** 2

        # Predicted volatility from GARCH model

        # Compute Metrics
        # Ensure both actual_volatility and garch_preds have the same length
        actual_volatility = actual_volatility.dropna()
        min_len = min(len(actual_volatility), len(garch_preds))
        garch_rmse = np.sqrt(mean_squared_error(actual_volatility[:min_len], garch_preds[:min_len]))
        garch_mae = mean_absolute_error(actual_volatility[:min_len], garch_preds[:min_len])
        garch_r2 = r2_score(actual_volatility[:min_len], garch_preds[:min_len])

        print(f"\n📊 GARCH Performance:")
        print(f"✅ RMSE: {garch_rmse}")
        print(f"✅ MAE: {garch_mae}")
        print(f"✅ R² Score: {garch_r2}")

        # Plot
        # Ensure plotting range matches the forecasted data
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index[:min_len], actual_volatility[:min_len], label="Actual Volatility")
        plt.plot(self.test.index[:min_len], volatility_preds[:min_len], linestyle="dashed", label="Predicted Volatility")
        plt.legend()
        plt.title("GARCH Volatility Prediction")
        plt.savefig(
            f"{base_dir}/notebooks/plots/garch_volatility_prediction.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

        print(self.garch_result.summary())

    def bayesian_inference(self):
        """Perform Bayesian analysis using PyMC3"""
        print(f"\n{'*'*100}\n")
        print("Performing Bayesian analysis using PyMC.\n")
        with pm.Model() as model:
            sigma = pm.Exponential("sigma", 1.0)
            mu = pm.Normal("mu", mu=self.train['Price'].mean(), sigma=10)
            likelihood = pm.Normal(
                "obs", mu=mu, sigma=sigma, observed=self.data['Price'])
            trace = pm.sample(1000, return_inferencedata=True)

        # Extract posterior mean from Bayesian inference
        posterior_mean = trace.posterior["mu"].mean(dim=("chain", "draw")).values

        # Compute Metrics
        bayesian_rmse = np.sqrt(mean_squared_error(self.test['Price'], [posterior_mean] * len(self.test)))
        bayesian_mae = mean_absolute_error(self.test['Price'], [posterior_mean] * len(self.test))
        bayesian_r2 = r2_score(self.test['Price'], [posterior_mean] * len(self.test))

        print(f"\n📊 Bayesian Inference Performance:")
        print(f"✅ RMSE: {bayesian_rmse}")
        print(f"✅ MAE: {bayesian_mae}")
        print(f"✅ R² Score: {bayesian_r2}")


        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index, self.test['Price'], label="Actual Price")
        plt.axhline(posterior_mean, color='r', linestyle="dashed", label="Bayesian Mean Prediction")
        plt.legend()
        plt.title("Bayesian Inference Prediction")
        plt.savefig(
            f"{base_dir}/notebooks/plots/bayesian_prediction.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

        print("Performing Bayesian analysis using PYMC completed!")

        # Print summary of posterior distributions
        print(f"\n{'-'*100}\n")
        print("\nPosterior Summary:")
        print(trace.posterior)  # Prints the raw posterior samples
        # Prints a statistical summary (mean, std, etc.)
        print(pm.summary(trace))

        # Print sample statistics
        print(f"\n{'-'*100}\n")
        print("\nSample Statistics:")
        print(trace.sample_stats)

        # Print observed data
        print(f"\n{'-'*100}\n")
        print("\nObserved Data:")
        print(trace.observed_data)

        return trace


    def compare_models(self):
        """Compare model performance using RMSE, MAE, and R² Score."""

        print(f"\n{'*'*100}\n")
        print("Comparing Model Performance\n")

        # --------------------- ARIMA Model Evaluation ---------------------
        arima_preds = self.arima_result.forecast(steps=len(self.test))
        arima_rmse = np.sqrt(mean_squared_error(self.test['Price'], arima_preds))
        arima_mae = mean_absolute_error(self.test['Price'], arima_preds)
        arima_r2 = r2_score(self.test['Price'], arima_preds)

        # --------------------- Bayesian Inference Evaluation ---------------------
        bayes_preds = [self.train['Price'].mean()] * len(self.test)
        bayesian_rmse = np.sqrt(mean_squared_error(self.test['Price'], bayes_preds))
        bayesian_mae = mean_absolute_error(self.test['Price'], bayes_preds)
        bayesian_r2 = r2_score(self.test['Price'], bayes_preds)

        # --------------------- GARCH Model Evaluation ---------------------
        actual_volatility = self.test['Price'].pct_change() ** 2  # Squared returns

        # Fix: Adjust start to be within the training data range, and align the lengths of arrays
        garch_forecast = self.garch_result.forecast(
            start=len(self.train) - len(self.test) + 1,  # Adjusted start to match the length of actual_volatility
            horizon=len(self.test) -1 # Adjusted horizon to match the length of actual_volatility
        )
        garch_preds = garch_forecast.variance.values[-1, :]  # Predicted volatility

        # Ensure both arrays have the same length
        min_len = min(len(actual_volatility.dropna()), len(garch_preds))
        actual_volatility = actual_volatility.dropna()[:min_len]
        garch_preds = garch_preds[:min_len]

        garch_rmse = np.sqrt(mean_squared_error(actual_volatility, garch_preds))
        garch_mae = mean_absolute_error(actual_volatility, garch_preds)
        garch_r2 = r2_score(actual_volatility, garch_preds)

        # --------------------- Print Model Comparison ---------------------
        metrics = pd.DataFrame({
            "Model": ["ARIMA", "GARCH", "Bayesian Inference"],
            "RMSE": [arima_rmse, garch_rmse, bayesian_rmse],
            "MAE": [arima_mae, garch_mae, bayesian_mae],
            "R² Score": [arima_r2, garch_r2, bayesian_r2]
        })

        print(metrics)

        print(f"\n{'-'*100}\n")
        print("Model with lowest RMSE and MAE is generally the best.")
        print("Higher R² Score (closer to 1) indicates better model fit.\n")
