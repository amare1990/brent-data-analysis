import numpy as np
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web

import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scripts.lstm import LSTMTimeSeries

import arviz as az

import os
from dotenv import load_dotenv

load_dotenv()

try:
    # Replace "API_KEY" with your actual environment variable name
    api_key = os.getenv("FRED_API_KEY")
    if api_key is None:
        raise ValueError("API key is not set in the environment variables.")
    print("API key loaded")
except ValueError as e:
    print(e)


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"


class ChangePointAnalysis:
    def __init__(self, processed_data_path, start_date, end_date):
        self.processed_data_path = processed_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(
            processed_data_path,
            parse_dates=["Date"],
            index_col="Date")
        # self.gpr = None  # Initialize gpr as None to handle missing data
        # properly
        self.merged_data = None  # Initialize merged_data to avoid AttributeError

    def download_data(self):
        print("Starting donwloading")
        print(f"✅ Data loaded from {self.processed_data_path}")
        print(self.data.columns)
        print(f"Index: {self.data.index}")
        print(f"{'-'*100}")
        print("Loading historical data completed!")
        # Download GDP, Inflation, USD Rate data from FRED
        self.gdp = pdr.DataReader(
            'GDP', 'fred', self.start_date, self.end_date)
        print("\nGDP data downloaded successfully!\n")
        print(self.gdp.columns)
        print(f"Index.self.gdp.index")
        print(f"{'-'*100}")
        self.inflation = pdr.DataReader(
            'CPIAUCSL', 'fred', self.start_date, self.end_date)
        print("\nInflation data downloaded successfully!\n")
        print(self.inflation.columns)
        print(f"Index: {self.inflation.index}")
        print(f"{'-'*100}")
        self.usd_rate = pdr.DataReader(
            'DEXUSEU', 'fred', self.start_date, self.end_date)
        print("\nUSD Rate data downloaded successfully!\n")
        print(self.usd_rate.columns)
        print(f"Index: {self.usd_rate.index}")
        print(f"{'-'*100}")

        # # Download Geopolitical Risk Index (GPR) from FRED using the API key
        # try:
        #     self.gpr = web.DataReader('GPR', 'fred', self.start_date, self.end_date, api_key=api_key)
        #     print("GPR data downloaded successfully!")
        # except Exception as e:
        #     print(f"Error downloading GPR data: {e}")
        #     self.gpr = None  # Set gpr to None if there is an error

    def merge_data(self):
        """Merges all datasets on the Date index and ensures row count consistency by filling missing values with the mean."""

        def ensure_datetime_index(df):
            """Ensures the dataframe index is a DateTimeIndex."""
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            return df

        # Ensure all datasets have Date as a datetime index
        self.data = ensure_datetime_index(self.data)
        self.gdp = ensure_datetime_index(self.gdp)
        self.inflation = ensure_datetime_index(self.inflation)
        self.usd_rate = ensure_datetime_index(self.usd_rate)

        # Merge all datasets using the Date index
        print("Merging data...")
        self.merged_data = self.data.join(self.gdp, how='left', rsuffix='_gdp')
        self.merged_data = self.merged_data.join(
            self.inflation, how='left', rsuffix='_inflation')
        self.merged_data = self.merged_data.join(
            self.usd_rate, how='left', rsuffix='_usd_rate')

        # Fill missing values using the mean of each column
        for col in ['GDP', 'Inflation_Rate',
                    'USD_Rate']:  # Adjust column names if different
            if col in self.merged_data.columns:
                self.merged_data[col].fillna(
                    self.merged_data[col].mean(), inplace=True)

        print("Data merged successfully!")
        print(self.merged_data.head())  # Debugging: Show first rows
        print(self.merged_data.columns)
        print(f"Index: {self.merged_data.index}")
        print(f"Shape: {self.merged_data.shape}")
        print(f"{'-'*100}")

        # Make sure Date is set as index
        # self.merged_data.set_index("Date", inplace=True)

        # Save merged file
        self.merged_data.to_csv(f"{base_dir}/data/merged_data.csv")
        print(f"Merged data saved as {base_dir}/merged_data.csv")

        return self.merged_data

    def scale_data(self):
        """Scale data to ensure no NaN values before scaling."""
        if self.merged_data is not None:  # Check if merged_data is available
            # Check for NaN values in the data
            if self.merged_data.isna().sum().sum() > 0:
                print("Missing values found. Handling missing values...")

                # Fill missing values with the mean of each column (or use
                # another method like median)
                self.merged_data.fillna(self.merged_data.mean(), inplace=True)
                print("Missing values filled with column mean.")
            else:
                print("No missing values found.")

            # Initialize the scaler
            scaler = StandardScaler()

            # Apply the scaler to the merged data (excluding the 'Date' index)
            self.scaled_data = scaler.fit_transform(self.merged_data)
            print("Data scaling complete.")
        else:
            print(
                "No merged data available. Please ensure merge_data is called successfully.")

    def split_data(self, train_ratio=0.8):
        """Split time series data into training and testing sets."""
        if self.merged_data is not None:
            train_size = int(len(self.merged_data) * train_ratio)
            return self.merged_data.iloc[:train_size], self.merged_data.iloc[train_size:]
        else:
            print("Merged data is not available. Please merge the data first.")
            return None, None

    def fit_var(self, lags=5):
        """Fit a Vector Autoregressive (VAR) model with predictions and performance metrics."""
        print(f"\n{'*'*100}\n")
        print("Fitting VAR model.\n")

        # Call split_data to create self.train and self.test
        self.train, self.test = self.split_data()
        # Fill NaN values
        self.train.fillna(self.train.mean(), inplace=True)

        # Alternatively, drop rows with NaN or infinite values
        self.train = self.train.dropna()
        self.train = self.train[~self.train.isin(
            [float('inf'), -float('inf')]).any(axis=1)]

        # Check if merged_data is not None
        if self.train is None:
            print("Error: merged_data is None. Please merge data first.")
            return

        # Ensure that the data used for VAR is stationary
        var_result = None  # Initialize var_result to None
        try:
            # Verify the columns are present in the train data
            required_columns = ['Price', 'CPIAUCSL', 'DEXUSEU', 'GDP']
            if not all(col in self.train.columns for col in required_columns):
                print("Error: Required columns are not present in the training data.")
                return

            # Initialize and fit the VAR model
            # Select the required columns
            var_model = VAR(self.train[required_columns])
            var_result = var_model.fit(lags)
        except Exception as e:
            print(f"Error during VAR model fitting: {e}")

        var_data = self.merged_data[required_columns]
        # Proceed only if var_result is not None
        if var_result is not None:
            # Forecast
            var_preds = var_result.forecast(
                var_data.values[-lags:], steps=len(self.test))

            # Compute Metrics
            # Assuming 'Price' is the first column
            var_rmse = np.sqrt(mean_squared_error(
                self.test['Price'], var_preds[:, 0]))
            var_mae = mean_absolute_error(self.test['Price'], var_preds[:, 0])
            var_r2 = r2_score(self.test['Price'], var_preds[:, 0])

            # Store performance metrics
            var_result_perf = {"rmse": var_rmse, "mae": var_mae, "r2": var_r2}

            print(f"📊 VAR Performance:")
            print(f"✅ RMSE: {var_rmse}")
            print(f"✅ MAE: {var_mae}")
            print(f"✅ R² Score: {var_r2}")

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.test.index, self.test['Price'], label="Actual Price")
            plt.plot(self.test.index,
                     var_preds[:,
                               0],
                     linestyle="dashed",
                     label="Predicted Price (VAR)")
            plt.legend()
            plt.title("VAR Predictions vs Actual")
            plt.savefig(
                f"{base_dir}/notebooks/plots/var_prediction.png",
                dpi=300,
                bbox_inches="tight")
            plt.show()

            print(var_result.summary())

            # Return var_result and var_result_perf
            return var_result, var_result_perf
        else:
            print(
                "VAR model fitting failed. Cannot proceed with predictions or performance evaluation.")
            return None, None

    def fit_markov_switching_arima(self, order=(
            1, 1, 1), n_regimes=2, train_ratio=0.8):
        """
        Fit a Markov-Switching ARIMA model, detect regime changes, make predictions, and compute performance metrics.
        """

        print("\n" + "*" * 100)
        print("Fitting Markov-Switching ARIMA model.\n")

        # Step 1: Ensure dataset is valid
        if self.merged_data is None or self.merged_data.empty:
            print("❌ Error: Dataset is empty or not loaded.")
            return None, None, None, None

        # Step 2: Ensure DateTime index
        self.merged_data.index = pd.to_datetime(self.merged_data.index)
        self.merged_data = self.merged_data.sort_index()  # Ensure chronological order

        # Step 3: Split the dataset manually (no `split_data()`)
        train_size = int(len(self.merged_data) * train_ratio)
        self.train = self.merged_data.iloc[:train_size]
        self.test = self.merged_data.iloc[train_size:]

        print(
            f"📊 Training set size: {len(self.train)}, Testing set size: {len(self.test)}")

        # Step 4: Check and handle missing values
        print(
            f"🔍 Missing values in train set: {self.train['Price'].isna().sum()}")
        print(
            f"🔍 Missing values in test set: {self.test['Price'].isna().sum()}")

        self.train['Price'].fillna(method='ffill', inplace=True)
        self.test['Price'].fillna(method='ffill', inplace=True)

        # Step 5: Set frequency explicitly (since frequency wasn't detected)
        if self.train.index.freq is None:
            print("📅 Detected Frequency: None. Setting frequency to quarterly.")
            self.train = self.train.asfreq('Q')
            self.test = self.test.asfreq('Q')

        # Step 6: Make Data Stationary (First-Order Differencing)
        self.train['Price_diff'] = self.train['Price'].diff().dropna()

        # Step 7: Fit the Markov-Switching AR Model
        try:
            model = MarkovAutoregression(
                self.train['Price_diff'].dropna(),  # Use differenced series
                k_regimes=n_regimes,
                order=order[0],
                switching_ar=False
            )
            result = model.fit()
        except Exception as e:
            print("❌ Error during model fitting:", e)
            return None, None, None, None

        # Step 8: Extract regime probabilities & states
        try:
            regime_probabilities = result.smoothed_marginal_probabilities
            predicted_states = regime_probabilities.idxmax(axis=1)
        except Exception as regime_e:
            print("❌ Error extracting regime probabilities:", regime_e)
            return None, None, None, None

        # Step 9: Make Predictions (Fix `start` and `end` values)
        try:
            # Explicitly handle start and end using iloc to ensure proper
            # indexing
            start_idx = self.test.index[0]
            end_idx = self.test.index[-1]

            # Use iloc to match the start and end
            markov_preds = result.predict(start=start_idx, end=end_idx)
            markov_preds = pd.Series(markov_preds, index=self.test.index)

            # Convert differenced predictions back to original scale
            markov_preds = self.train['Price'].iloc[-1] + markov_preds.cumsum()

            print(f"✅ First 5 predictions:\n{markov_preds.head()}")
        except Exception as pred_e:
            print("⚠️ Error during prediction:", pred_e)
            return None, None, None, None

        # Step 10: Check for NaNs in predictions
        if markov_preds.isna().sum() > 0:
            print(
                "❌ Error: NaN values detected in predictions. Aborting metric calculation.")
            print("🛑 Debugging NaN Predictions:")
            print(markov_preds)
            return None, None, None, None

        # Step 11: Compute Performance Metrics
        try:
            markov_rmse = np.sqrt(
                mean_squared_error(
                    self.test['Price'],
                    markov_preds))
            markov_mae = mean_absolute_error(self.test['Price'], markov_preds)
            markov_r2 = r2_score(self.test['Price'], markov_preds)
        except Exception as metric_e:
            print("❌ Error computing performance metrics:", metric_e)
            return None, None, None, None

        markov_result_perf = {
            "rmse": markov_rmse,
            "mae": markov_mae,
            "r2": markov_r2}
        print(f"\n📊 Markov-Switching ARIMA Performance:")
        print(f"✅ RMSE: {markov_rmse:.4f}")
        print(f"✅ MAE: {markov_mae:.4f}")
        print(f"✅ R² Score: {markov_r2:.4f}\n")

        # Step 12: Plot Predictions vs Actual
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.test.index,
            self.test['Price'],
            label="Actual Price",
            color="blue")
        plt.plot(
            self.test.index,
            markov_preds,
            linestyle="dashed",
            label="Predicted Price (MS-AR)",
            color="red")
        plt.legend()
        plt.title("Markov-Switching ARIMA Predictions vs Actual")
        plt.savefig(
            f"{base_dir}/notebooks/plots/preds_vs_actual_markov_switching.png",
            dpi=300,
            bbox_inches="tight")
        plt.show()

        # Step 13: Plot Regime Switching
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.train.index,
            self.train['Price'],
            label="Price",
            color='black')
        plt.scatter(
            self.train.index,
            self.train['Price'],
            c=predicted_states,
            cmap="coolwarm",
            label="Regime")
        plt.title("Markov-Switching ARIMA - Regime Detection")
        plt.legend()
        plt.savefig(
            f"{base_dir}/notebooks/plots/regime_changing_markov_switching.png",
            dpi=300,
            bbox_inches="tight")
        plt.show()

        print(result.summary())
        return markov_preds, markov_result_perf, regime_probabilities, predicted_states

    def fit_lstm(self, epochs=5, batch_size=32):
        """Fit an LSTM model for time series forecasting."""
        print(f"\n{'*'*100}\n")
        # print("Fitting LSTM model.\n")

        # Initialize the LSTMTimeSeries with the data, epochs, and batch size
        print("I am inside fit_lstm in change data nalysis")
        print(f"Merged data first five rows:\n {self.merged_data.head()}")
        lstm_model = LSTMTimeSeries(
            data=self.merged_data,
            epochs=epochs,
            batch_size=batch_size)

        # Fit the model
        model, perf_metrics = lstm_model.fit()

        print("Fitting lstm model completed successfully!")

        # Optionally, return the trained model if you need it for further
        # predictions or analysis
        return model, perf_metrics

    def compare_models(self):
        """Compare model performance using RMSE, MAE, and R² Score."""

        print(f"\n{'*'*100}\n")
        print("Comparing Model Performance\n")

        # Fit the models and get their performance metrics
        _, var_result_perf = self.fit_var()
        # _, markov_result_perf = self.fit_markov_switching_arima()
        # markov_preds, markov_result_perf, regime_probabilities, predicted_states = self.fit_markov_switching_arima()
        # print(f"✅ Markov-Switching ARIMA Performance: {markov_result_perf}")
        # print(f"✅ Regime Probabilities: {regime_probabilities}")
        # print(f"✅ Predicted States: {predicted_states}")
        # print(f"✅ Markov-Switching ARIMA Predictions: {markov_preds}")

        # --------------------- LSTM Model Evaluation ---------------------
        lstm_model = LSTMTimeSeries(
            data=self.merged_data, epochs=5, batch_size=32)

        lstm_model, perf_metrics = lstm_model.fit()
        lstm_rmse = perf_metrics['rmse']
        lstm_mae = perf_metrics['mae']
        lstm_r2 = perf_metrics['r2']
        print(f"LSTM Performance Metrics:")
        print(f"RMSE: {perf_metrics['rmse']}")
        print(f"MAE: {perf_metrics['mae']}")
        print(f"R² Score: {perf_metrics['r2']}")

        # if var_result_perf is None or markov_result_perf is None or lstm_rmse
        # is None:
        if var_result_perf is None or lstm_rmse is None:
            print("⚠️ One or more models failed. Cannot compute comparison metrics.")
            return

        # Create a DataFrame to store the performance metrics
        # metrics = pd.DataFrame({
        #     "Model": ["VAR", "Markov-Switching ARIMA", 'LSTM'],
        #     "RMSE": [var_result_perf["rmse"], markov_result_perf["rmse"], lstm_rmse],
        #     "MAE": [var_result_perf["mae"], markov_result_perf["mae"], lstm_mae],
        #     "R² Score": [var_result_perf["r2"], markov_result_perf["r2"], lstm_r2]
        # })
        metrics = pd.DataFrame({
            "Model": ["VAR", 'LSTM'],
            "RMSE": [var_result_perf["rmse"], lstm_rmse],
            "MAE": [var_result_perf["mae"], lstm_mae],
            "R² Score": [var_result_perf["r2"], lstm_r2]
        })

        print(metrics)

        print(f"\n{'-'*100}\n")
        print("Model with lowest RMSE and MAE is generally the best.")
        print("Higher R² Score (closer to 1) indicates better model fit.\n")
