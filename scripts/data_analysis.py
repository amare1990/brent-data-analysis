"""Data Analysis Workflow."""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class BrentDataAnalysis:
    def __init__(self, file_path):
        """Initialize with data file path"""
        self.file_path = file_path
        self.data = self.load_data()
        # self.load_data = None

    def load_data(self):
        """Load Brent oil price dataset"""
        self.data = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")
        print(f"\n{'*'*70}")
        print("✅ Data loaded successfully!")
        return self.data

    def overview_of_data(self):
        """Provide an overview of the dataset."""
        print(f"\n{'*'*70}")
        print("Overview of the Data:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())

    def describe_data(self):
        """Provide basic summary statistics"""
        print(f"\n{'*'*70}")
        print(f"Basic statistical summary for numerical columns\n {self.data.describe()}")
        if self.data.dtypes[self.data.dtypes == "object"].any():
          print(f"\nBasic statistical summary for categorical columns\n {self.data.describe(include=[object, 'category'])}")
        else:
          print("No categorical or object columns found.")

    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        print(f"\n{'*'*70}")
        print("Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n")

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        rows_before = self.data.shape[0]
        self.data.drop_duplicates(inplace=True)
        rows_after = self.data.shape[0]
        print(f"\n{'*'*70}")
        print("Duplicates removed.")
        print(f"Number of rows removed for the duplicate rows case: {rows_before - rows_after}")
        print(f"Total rows left after duplicate rows removed. \n {self.data.shape[0]}")
        print("\n")

    def plot_time_series(self):
        """Plot time series data."""
        print(f"\n{'*'*70}")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.data, x=self.data.index, y='Price')
        plt.title('Brent Oil Price Time Series')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../notebooks/plots/ctime_series_plot.png')
        plt.show()


