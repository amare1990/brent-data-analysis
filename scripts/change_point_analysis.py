"""Change Point Analysis."""

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web



import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Now you can access the API key using os.getenv()
api_key = os.getenv('FRED_API_KEY')

if api_key:
    print("API Key loaded successfully.")
else:
    print("API Key not found. Please check your .env file.")





class ChangePointAnalysis:
    def __init__(self, processed_data_path, start_date, end_date):
        self.processed_data_path = processed_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(processed_data_path, parse_dates=["Date"], index_col="Date")

    def download_data(self):
        # Fetch API Key from environment variable
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("API key is missing. Please set the FRED_API_KEY environment variable.")

        # Download GDP, Inflation, USD Rate data from FRED
        self.gdp = pdr.DataReader('GDP', 'fred', self.start_date, self.end_date)
        self.inflation = pdr.DataReader('CPIAUCSL', 'fred', self.start_date, self.end_date)
        self.usd_rate = pdr.DataReader('DEXUSEU', 'fred', self.start_date, self.end_date)

        # Download Geopolitical Risk Index (GPR) from FRED using the API key
        try:
            self.gpr = web.DataReader('GPR', 'fred', self.start_date, self.end_date, api_key=api_key)
            print("GPR data downloaded successfully!")
        except Exception as e:
            print(f"Error downloading GPR data: {e}")
