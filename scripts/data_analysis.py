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

    def describe_data(self):
        """Provide basic summary statistics"""
        print(f"Basic statistical summary for numerical columns\n {self.data.describe()}")
        print(f"\nBasic statistical summary for categorical columns\n {self.data.describe(include=[object, 'category'])}")
