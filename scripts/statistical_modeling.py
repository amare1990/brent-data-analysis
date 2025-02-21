"""Statistical modeling."""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc3 as pm



class StatisticalModel:
    def __init__(self, data):
        self.data = data
