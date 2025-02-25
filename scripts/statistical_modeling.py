"""Statistical modeling."""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.api import VAR
# from statsmodels.tsa.regime_switching import MarkovRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc as pm
import arviz as az

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scripts.lstm import LSTMTimeSeries


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
        print(f"\n{'*'*100}\n")
        print("Checking for stationarity using ADF test.")
        result = adfuller(self.data['Price'])
        return {'ADF Statistic': result[0], 'p-value': result[1]}

    def plot_acf_pacf(self, lags=30):
        """Plot ACF and PACF to determine AR and MA orders."""
        print(f"\n{'*'*100}\n")
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
        arima_rmse = np.sqrt(
            mean_squared_error(
                self.test['Price'],
                arima_preds))
        arima_mae = mean_absolute_error(self.test['Price'], arima_preds)
        arima_r2 = r2_score(self.test['Price'], arima_preds)

        print(f"📊 ARIMA Performance:")
        print(f"✅ RMSE: {arima_rmse}")
        print(f"✅ MAE: {arima_mae}")
        print(f"✅ R² Score: {arima_r2}")

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index, self.test['Price'], label="Actual Price")
        plt.plot(
            self.test.index,
            arima_preds,
            linestyle="dashed",
            label="Predicted Price")
        plt.legend()
        plt.title("ARIMA Predictions vs Actual")
        plt.savefig(
            f"{base_dir}/notebooks/plots/arima_prediction.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        print(self.arima_result.summary())
        return self.arima_result.summary()

    def fit_garch(self):
        """Fit a GARCH model"""
        print(f"\n{'*'*100}\n")
        print("Fitting GARCH model.\n")
        garch = arch_model(
            self.train['Price'].pct_change().dropna(),
            vol='Garch',
            p=1,
            q=1)
        self.garch_result = garch.fit()

        # Forecast volatility
        # Adjust start and horizon to ensure they are within the data range
        # Limit horizon to training data size
        horizon = min(len(self.test), len(self.train))
        start = len(self.train) - horizon
        garch_forecast = self.garch_result.forecast(
            start=start, horizon=horizon)

        volatility_preds = garch_forecast.variance.values[-1, :]
        garch_preds = volatility_preds

        # Actual squared returns
        actual_volatility = self.test['Price'].pct_change() ** 2

        # Predicted volatility from GARCH model

        # Compute Metrics
        # Ensure both actual_volatility and garch_preds have the same length
        actual_volatility = actual_volatility.dropna()
        min_len = min(len(actual_volatility), len(garch_preds))
        garch_rmse = np.sqrt(mean_squared_error(
            actual_volatility[:min_len], garch_preds[:min_len]))
        garch_mae = mean_absolute_error(
            actual_volatility[:min_len], garch_preds[:min_len])
        garch_r2 = r2_score(actual_volatility[:min_len], garch_preds[:min_len])

        print(f"\n📊 GARCH Performance:")
        print(f"✅ RMSE: {garch_rmse}")
        print(f"✅ MAE: {garch_mae}")
        print(f"✅ R² Score: {garch_r2}")

        # Plot
        # Ensure plotting range matches the forecasted data
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index[:min_len],
                 actual_volatility[:min_len],
                 label="Actual Volatility")
        plt.plot(self.test.index[:min_len],
                 volatility_preds[:min_len],
                 linestyle="dashed",
                 label="Predicted Volatility")
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
        posterior_mean = trace.posterior["mu"].mean(
            dim=("chain", "draw")).values

        # Compute Metrics
        bayesian_rmse = np.sqrt(mean_squared_error(
            self.test['Price'], [posterior_mean] * len(self.test)))
        bayesian_mae = mean_absolute_error(
            self.test['Price'], [posterior_mean] * len(self.test))
        bayesian_r2 = r2_score(self.test['Price'], [
                               posterior_mean] * len(self.test))

        print(f"\n📊 Bayesian Inference Performance:")
        print(f"✅ RMSE: {bayesian_rmse}")
        print(f"✅ MAE: {bayesian_mae}")
        print(f"✅ R² Score: {bayesian_r2}")

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index, self.test['Price'], label="Actual Price")
        plt.axhline(
            posterior_mean,
            color='r',
            linestyle="dashed",
            label="Bayesian Mean Prediction")
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

    # def fit_var(self, lags=5):
    #     """Fit a Vector Autoregressive (VAR) model with predictions and performance metrics."""
    #     print(f"\n{'*'*100}\n")
    #     print("Fitting VAR model.\n")

    #     # Ensure that the data used for VAR is stationary
    #     var_data = self.data[['Price', 'OtherFeature']]  # Add more features as needed
    #     var_model = VAR(var_data)
    #     var_result = var_model.fit(lags)

    #     # Forecast
    #     var_preds = var_result.forecast(var_data.values[-lags:], steps=len(self.test))

    #     # Compute Metrics
    #     var_rmse = np.sqrt(mean_squared_error(self.test['Price'], var_preds[:, 0]))  # Assuming 'Price' is the first column
    #     var_mae = mean_absolute_error(self.test['Price'], var_preds[:, 0])
    #     var_r2 = r2_score(self.test['Price'], var_preds[:, 0])

    #     print(f"📊 VAR Performance:")
    #     print(f"✅ RMSE: {var_rmse}")
    #     print(f"✅ MAE: {var_mae}")
    #     print(f"✅ R² Score: {var_r2}")

    #     # Plot
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.test.index, self.test['Price'], label="Actual Price")
    #     plt.plot(self.test.index, var_preds[:, 0], linestyle="dashed", label="Predicted Price (VAR)")
    #     plt.legend()
    #     plt.title("VAR Predictions vs Actual")
    #     plt.savefig(f"{base_dir}/notebooks/plots/var_prediction.png", dpi=300, bbox_inches="tight")
    #     plt.show()

    #     print(var_result.summary())
    #     return var_result

    # def fit_markov_switching_arima(self, order=(1, 1, 1), n_regimes=2):
    #     """Fit a Markov-Switching ARIMA model with predictions and performance metrics."""
    #     print(f"\n{'*'*70}\n")
    #     print("Fitting Markov-Switching ARIMA model.\n")

    #     # Fit the Markov-Switching ARIMA model
    #     model = MarkovRegression(self.data['Price'], k_regimes=n_regimes, order=order)
    #     result = model.fit()

    #     # Forecast
    #     markov_preds = result.predict(start=self.test.index[0], end=self.test.index[-1])

    #     # Compute Metrics
    #     markov_rmse = np.sqrt(mean_squared_error(self.test['Price'], markov_preds))
    #     markov_mae = mean_absolute_error(self.test['Price'], markov_preds)
    #     markov_r2 = r2_score(self.test['Price'], markov_preds)

    #     print(f"📊 Markov-Switching ARIMA Performance:")
    #     print(f"✅ RMSE: {markov_rmse}")
    #     print(f"✅ MAE: {markov_mae}")
    #     print(f"✅ R² Score: {markov_r2}")

    #     # Plot
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.test.index, self.test['Price'], label="Actual Price")
    #     plt.plot(self.test.index, markov_preds, linestyle="dashed", label="Predicted Price (Markov-Switching ARIMA)")
    #     plt.legend()
    #     plt.title("Markov-Switching ARIMA Predictions vs Actual")
    #     plt.savefig(f"{base_dir}/notebooks/plots/markov_switching_arima_prediction.png", dpi=300, bbox_inches="tight")
    #     plt.show()

    #     print(result.summary())
    #     return result

    def fit_lstm(self, epochs=100, batch_size=32):
        """Fit an LSTM model for time series forecasting."""
        print(f"\n{'*'*100}\n")
        # print("Fitting LSTM model.\n")

        # Initialize the LSTMTimeSeries with the data, epochs, and batch size
        lstm_model = LSTMTimeSeries(
            data=self.data,
            epochs=epochs,
            batch_size=batch_size)

        # Fit the model
        model, perf_metrics = lstm_model.fit()

        # Optionally, return the trained model if you need it for further
        # predictions or analysis
        return model, perf_metrics

    def compare_models(self):
        """Compare model performance using RMSE, MAE, and R² Score."""

        print(f"\n{'*'*100}\n")
        print("Comparing Model Performance\n")

        # --------------------- ARIMA Model Evaluation ---------------------
        arima_preds = self.arima_result.forecast(steps=len(self.test))
        arima_rmse = np.sqrt(
            mean_squared_error(
                self.test['Price'],
                arima_preds))
        arima_mae = mean_absolute_error(self.test['Price'], arima_preds)
        arima_r2 = r2_score(self.test['Price'], arima_preds)

        # --------------------- Bayesian Inference Evaluation -----------------
        bayes_preds = [self.train['Price'].mean()] * len(self.test)
        bayesian_rmse = np.sqrt(
            mean_squared_error(
                self.test['Price'],
                bayes_preds))
        bayesian_mae = mean_absolute_error(self.test['Price'], bayes_preds)
        bayesian_r2 = r2_score(self.test['Price'], bayes_preds)

        # --------------------- GARCH Model Evaluation ---------------------
        # Squared returns
        actual_volatility = self.test['Price'].pct_change() ** 2
        garch_forecast = self.garch_result.forecast(
            start=len(self.train) - len(self.test),
            horizon=len(self.test)
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
        lstm_model = LSTMTimeSeries(data=self.data, epochs=10, batch_size=32)

        lstm_model, perf_metrics = lstm_model.fit()
        lstm_rmse = perf_metrics['rmse']
        lstm_mae = perf_metrics['mae']
        lstm_r2 = perf_metrics['r2']
        print(f"LSTM Performance Metrics:")
        print(f"RMSE: {perf_metrics['rmse']}")
        print(f"MAE: {perf_metrics['mae']}")
        print(f"R² Score: {perf_metrics['r2']}")

        # print(f"📊 LSTM Performance:")
        # print(f"✅ RMSE: {rmse}")
        # print(f"✅ MAE: {mae}")
        # print(f"✅ R² Score: {r2}")

        # --------------------- Print Model Comparison ---------------------
        metrics = pd.DataFrame({
            "Model": ["ARIMA", "GARCH", "Bayesian Inference", "LSTM"],
            "RMSE": [arima_rmse, garch_rmse, bayesian_rmse, lstm_rmse],
            "MAE": [arima_mae, garch_mae, bayesian_mae, lstm_mae],
            "R² Score": [arima_r2, garch_r2, bayesian_r2, lstm_r2]
        })

        print(metrics.to_string(index=False))

        print(f"\n{'-'*100}\n")
        print("✅ Model with lowest RMSE and MAE is generally the best.")
        print("✅ Higher R² Score (closer to 1) indicates better model fit.\n")

    # def compare_models(self):
    #     """Compare model performance using RMSE, MAE, and R² Score."""

    #     print(f"\n{'*'*100}\n")
    #     print("Comparing Model Performance\n")

    #     # --------------------- ARIMA Model Evaluation ---------------------
    #     arima_preds = self.arima_result.forecast(steps=len(self.test))
    #     arima_rmse = np.sqrt(mean_squared_error(self.test['Price'], arima_preds))
    #     arima_mae = mean_absolute_error(self.test['Price'], arima_preds)
    #     arima_r2 = r2_score(self.test['Price'], arima_preds)

    #     # --------------------- Bayesian Inference Evaluation --------------
    #     bayes_preds = [self.train['Price'].mean()] * len(self.test)
    #     bayesian_rmse = np.sqrt(mean_squared_error(self.test['Price'], bayes_preds))
    #     bayesian_mae = mean_absolute_error(self.test['Price'], bayes_preds)
    #     bayesian_r2 = r2_score(self.test['Price'], bayes_preds)

    #     # --------------------- GARCH Model Evaluation ---------------------
    # actual_volatility = self.test['Price'].pct_change() ** 2  # Squared
    # returns

    #     # Fix: Adjust start to be within the training data range, and align the lengths of arrays
    #     garch_forecast = self.garch_result.forecast(
    #         start=len(self.train) - len(self.test) + 1,  # Adjusted start to match the length of actual_volatility
    #         horizon=len(self.test) -1 # Adjusted horizon to match the length of actual_volatility
    #     )
    # garch_preds = garch_forecast.variance.values[-1, :]  # Predicted
    # volatility

    #     # Ensure both arrays have the same length
    #     min_len = min(len(actual_volatility.dropna()), len(garch_preds))
    #     actual_volatility = actual_volatility.dropna()[:min_len]
    #     garch_preds = garch_preds[:min_len]

    #     garch_rmse = np.sqrt(mean_squared_error(actual_volatility, garch_preds))
    #     garch_mae = mean_absolute_error(actual_volatility, garch_preds)
    #     garch_r2 = r2_score(actual_volatility, garch_preds)

    #     # --------------------- VAR Model Evaluation ---------------------
    #     # Assuming fit_var returns the predicted values for the test set
    #     # var_preds = self.fit_var()  # Adjust based on your implementation
    #     # var_rmse = np.sqrt(mean_squared_error(self.test['Price'], var_preds))
    #     # var_mae = mean_absolute_error(self.test['Price'], var_preds)
    #     # var_r2 = r2_score(self.test['Price'], var_preds)

    #     # --------------------- Markov-Switching ARIMA Model Evaluation ----
    #     # Assuming fit_markov_switching_arima returns the predicted values for the test set
    #     # msarima_preds = self.fit_markov_switching_arima()  # Adjust based on your implementation
    #     # msarima_rmse = np.sqrt(mean_squared_error(self.test['Price'], msarima_preds))
    #     # msarima_mae = mean_absolute_error(self.test['Price'], msarima_preds)
    #     # msarima_r2 = r2_score(self.test['Price'], msarima_preds)

    #     # --------------------- LSTM Model Evaluation ---------------------
    #     lstm_model = LSTMModel(data=self.data)  # Initialize LSTM model
    # trained_lstm_model = lstm_model.fit_lstm(epochs=25, batch_size=32)  #
    # Train LSTM model

    #     # Get LSTM predictions from the trained model
    # lstm_preds = trained_lstm_model.predict(self.test_data)  # Adjust with
    # actual method to get predictions

    #     # Rescale predictions back to original values
    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    #     lstm_preds_rescaled = scaler.inverse_transform(lstm_preds.reshape(-1, 1))

    #     # Calculate RMSE, MAE, and R² score for LSTM model
    #     lstm_rmse = np.sqrt(mean_squared_error(self.test['Price'], lstm_preds_rescaled))
    #     lstm_mae = mean_absolute_error(self.test['Price'], lstm_preds_rescaled)
    #     lstm_r2 = r2_score(self.test['Price'], lstm_preds_rescaled)

    #     print(f"LSTM Performance Metrics:")
    #     print(f"RMSE: {lstm_rmse}")
    #     print(f"MAE: {lstm_mae}")
    #     print(f"R² Score: {lstm_r2}")

    #     # --------------------- Print Model Comparison ---------------------
    #     metrics = pd.DataFrame({
    #         "Model": ["ARIMA", "GARCH", "Bayesian Inference", "LSTM"],
    #         "RMSE": [arima_rmse, garch_rmse, bayesian_rmse, lstm_rmse],
    #         "MAE": [arima_mae, garch_mae, bayesian_mae, lstm_mae],
    #         "R² Score": [arima_r2, garch_r2, bayesian_r2, lstm_r2]
    #     })

    #     print(metrics)

    #     print(f"\n{'-'*100}\n")
    #     print("Model with lowest RMSE and MAE is generally the best.")
    #     print("Higher R² Score (closer to 1) indicates better model fit.\n")
