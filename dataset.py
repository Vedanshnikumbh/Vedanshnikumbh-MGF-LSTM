import os
import pandas as pd
import numpy as np
import yfinance as yf
import time
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, start_date="2019-01-01", end_date="2025-01-01", data_folder="data"):
        """
        Initializes the dataset handler with date ranges and data storage paths.
        
        Args:
            start_date (str): The start date for data fetching in YYYY-MM-DD format.
            end_date (str): The end date for data fetching in YYYY-MM-DD format.
            data_folder (str): The local directory name to store downloaded CSV files.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        self.all_symbols = ['^NSEI', '^INDIAVIX']

    def fetch_data(self, ticker):
        """
        Downloads historical price data for a specific ticker using yfinance.
        
        Args:
            ticker (str): The symbol to download (e.g., '^NSEI').
            
        Returns:
            pd.DataFrame: A DataFrame containing the downloaded OHLCV data.
        """
        print(f"   Downloading {ticker}...")
        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df.dropna(inplace=True)
            return df
        
        except Exception as e:
            print(f"   Error downloading {ticker}: {e}")
            return pd.DataFrame()

    def _ensure_data_exists(self):
        """
        Verifies local CSV availability and downloads missing data if necessary.
        """
        print("Checking local data availability...")
        for symbol in self.all_symbols:
            clean_name = symbol.replace('^', '')
            file_path = os.path.join(self.data_folder, f"{clean_name}.csv")
            if not os.path.exists(file_path):
                print(f"File {clean_name}.csv missing. Downloading...")
                df = self.fetch_data(symbol)
                if not df.empty:
                    df.to_csv(file_path)
                    print(f"   Saved {clean_name}.csv")
                time.sleep(0.5)
        print("Required CSV files present locally.")

    def load_csv(self, symbol):
        """
        Loads a symbol's data from a local CSV file.
        
        Args:
            symbol (str): The ticker symbol to load.
            
        Returns:
            pd.DataFrame: Loaded data or an empty DataFrame if the file is missing.
        """
        clean_name = symbol.replace('^', '')
        file_path = os.path.join(self.data_folder, f"{clean_name}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        return pd.DataFrame()

    def get_price_data(self):
        """
        Orchestrates data loading and aligns multiple sources by common dates.
        
        Returns:
            dict: A dictionary containing aligned DataFrames for 'NIFTY50' and 'VIX'.
        """
        self._ensure_data_exists()
        
        print("\nLoading and aligning data...")
        temp_storage = {}
        all_indices = []
        for symbol in self.all_symbols:
            df = self.load_csv(symbol)
            if df.empty:
                raise ValueError(f"CRITICAL: Could not load data for {symbol}.")
            temp_storage[symbol] = df
            all_indices.append(df.index)
            
        # Find the intersection of all dates present in NIFTY and VIX
        common_dates = reduce(pd.Index.intersection, all_indices)
        print(f"Alignment Complete. Common Date Count: {len(common_dates)} days.")
        
        if len(common_dates) == 0:
            raise ValueError("No common dates found! Check date ranges.")

        master_dict = {
            'NIFTY50': temp_storage['^NSEI'].loc[common_dates],
            'VIX': temp_storage['^INDIAVIX'].loc[common_dates]
        }
        
        return master_dict

if __name__ == "__main__":
    dataset = Dataset()
    try:
        data = dataset.get_price_data()
        print("\n" + "="*60)
        print("DATASET DIAGNOSTICS")
        print("="*60)
        for name, df in data.items():
            print(f"{name}: {len(df)} rows | Start: {df.index.min().date()} | End: {df.index.max().date()}")
        print("="*60)

    except Exception as e:
        print(f"\nCRITICAL FAILURE WHILE LOADING DATA: {e}")