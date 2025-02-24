"""Change Point Analysis."""

import numpy as np
import pandas as pd




class ChangePointAnalysis:
    def __init__(self, processed_data_path, start_date, end_date):
        self.processed_data_path = processed_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(processed_data_path, parse_dates=["Date"], index_col="Date")
