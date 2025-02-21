"""Statistical modeling."""

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc3 as pm



class StatisticalModel:
    def __init__(self, data):
        self.data = data


    def check_stationarity(self):
        result = adfuller(self.data['Price'])
        return {'ADF Statistic': result[0], 'p-value': result[1]}

    def plot_acf_pacf(self, lags=30):
        """Plot ACF and PACF to determine AR and MA orders."""
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Plot ACF (AutoCorrelation Function)
        plot_acf(self.data['Price'], lags=lags, ax=ax[0])
        ax[0].set_title("Autocorrelation Function (ACF)")

        # Plot PACF (Partial AutoCorrelation Function)
        plot_pacf(self.data['Price'], lags=lags, ax=ax[1])
        ax[1].set_title("Partial Autocorrelation Function (PACF)")

        plt.show()

    def fit_arima(self, order=(1,1,1)):
        """Fit an ARIMA model"""
        model = ARIMA(self.data['Price'], order=order)
        self.arima_result = model.fit()
        print(self.arima_result.summary())

    def fit_garch(self):
        """Fit a GARCH model"""
        from arch import arch_model
        garch = arch_model(self.data['Price'], vol='Garch', p=1, q=1)
        self.garch_result = garch.fit()
        print(self.garch_result.summary())

    def bayesian_inference(self):
        """Perform Bayesian analysis using PyMC3"""
        import pymc3 as pm
        with pm.Model() as model:
            sigma = pm.Exponential("sigma", 1.0)
            mu = pm.Normal("mu", mu=self.data['Price'].mean(), sigma=10)
            likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data['Price'])
            trace = pm.sample(1000, return_inferencedata=True)
        return trace
