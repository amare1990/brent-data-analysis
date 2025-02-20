"""Data Analysis Workflow."""

import pandas as pd


class BrentDataAnalysis:
    def __init__(self, file_path):
        """Initialize with data file path"""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load Brent oil price dataset"""
        self.df = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")
        print("✅ Data loaded successfully!")
        return self.df.head()
