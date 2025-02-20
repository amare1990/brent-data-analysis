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

    def overview_of_data(self):
        """Provide an overview of the dataset."""
        print("Overview of the Data:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())

    def describe_data(self):
        """Provide basic summary statistics"""
        print(f"Basic statistical summary for numerical columns\n {self.data.describe()}")
        print(f"\nBasic statistical summary for categorical columns\n {self.data.describe(include=[object, 'category'])}")
