"""Statistical modeling."""

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc as pm


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"



class StatisticalModel:
    def __init__(self, data):
        self.data = data


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

    def fit_arima(self, order=(1,1,1)):
        """Fit an ARIMA model"""
        print(f"\n{'*'*70}\n")
        print("Fitting ARIMA model.\n")
        model = ARIMA(self.data['Price'], order=order)
        self.arima_result = model.fit()
        print
        print(self.arima_result.summary())

    def fit_garch(self):
        """Fit a GARCH model"""
        print(f"\n{'*'*70}\n")
        print("Fitting GARCH model.\n")
        garch = arch_model(self.data['Price'], vol='Garch', p=1, q=1)
        self.garch_result = garch.fit()
        print(self.garch_result.summary())

    def bayesian_inference(self):
        """Perform Bayesian analysis using PyMC3"""
        print(f"\n{'*'*70}\n")
        print("Performing Bayesian analysis using PyMC.\n")
        with pm.Model() as model:
            sigma = pm.Exponential("sigma", 1.0)
            mu = pm.Normal("mu", mu=self.data['Price'].mean(), sigma=10)
            likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data['Price'])
            trace = pm.sample(1000, return_inferencedata=True)

        print("Performing Bayesian analysis using PYMC completed!")

        # Print summary of posterior distributions
        print(f"\n{'-'*70}\n")
        print("\nPosterior Summary:")
        print(trace.posterior)  # Prints the raw posterior samples
        print(pm.summary(trace))  # Prints a statistical summary (mean, std, etc.)

        # Print sample statistics
        print(f"\n{'-'*70}\n")
        print("\nSample Statistics:")
        print(trace.sample_stats)

        # Print observed data
        print(f"\n{'-'*70}\n")
        print("\nObserved Data:")
        print(trace.observed_data)

        return trace
