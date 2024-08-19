import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import traceback
from pyarrow import parquet as pq

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, data_dir='data', processed_dir='processed_data'):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.error_files = []
        self.symbol_map = {
            'MESH5': 'MES.FUT',  # March 2025 contract
            'MESM5': 'MES.FUT',  # June 2025 contract
            'MESU4': 'MES.FUT',  # September 2024 contract
            'MESU4-MESH5': 'MES.FUT',  # Spread, map to front-month (MESU4)
            'MESU4-MESZ4': 'MES.FUT',  # Spread, map to front-month (MESU4)
            'MESU5': 'MES.FUT',  # September 2025 contract
            'MESZ4': 'MES.FUT',  # December 2024 contract
            'MESZ4-MESH5': 'MES.FUT',  # Spread, map to front-month (MESZ4)
            'MNQH5': 'MNQ.FUT',  # March 2025 contract
            'MNQM5': 'MNQ.FUT',  # June 2025 contract
            'MNQU4': 'MNQ.FUT',  # September 2024 contract
            'MNQU4-MNQH5': 'MNQ.FUT',  # Spread, map to front-month (MNQU4)
            'MNQU4-MNQZ4': 'MNQ.FUT',  # Spread, map to front-month (MNQU4)
            'MNQU5': 'MNQ.FUT',  # September 2025 contract
            'MNQZ4': 'MNQ.FUT',  # December 2024 contract
        }
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_files(self, start_date, end_date, symbols):
        """
        Generator that yields processed data from files within the date range.

        Args:
            start_date (str or datetime): Start date for data loading.
            end_date (str or datetime): End date for data loading.
            symbols (list): List of symbols to load data for.

        Yields:
            pd.DataFrame: Processed and filtered data from each file.
        """
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)
        
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.ohlcv-1m.json')])
        
        for file in tqdm(files, desc="Processing files"):
            try:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_json(file_path, lines=True)
                
                logging.info(f"Processing file: {file}")
                logging.info(f"Initial shape: {df.shape}")
                
                # Extract timestamp and set as index
                df['timestamp'] = pd.to_datetime(df['hd'].apply(lambda x: x['ts_event']), utc=True)
                df = df.set_index('timestamp')
                
                logging.info(f"Date range in file: {df.index.min()} to {df.index.max()}")
                logging.info(f"Unique symbols before filtering: {df['symbol'].unique()}")
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                logging.info(f"Shape after date filtering: {df.shape}")
                
                if df.empty:
                    logging.info("No data left after date filtering.")
                    continue
                
                # Filter by symbols if specified
                if symbols != ["ALL_SYMBOLS"]:
                    df = df[df['symbol'].isin(symbols)]
                
                logging.info(f"Shape after symbol filtering: {df.shape}")
                logging.info(f"Unique symbols in filtered data: {df['symbol'].unique()}")
                
                if df.empty:
                    logging.info("No data left after symbol filtering.")
                    continue
                
                # Process numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df = df.drop('hd', axis=1)
                
                yield df
                
            except Exception as e:
                self.error_files.append((file, str(e)))
                logging.error(f"Error processing file {file}: {str(e)}")
                logging.debug(traceback.format_exc())
                
    def resample_data(self, df, timeframe):
        """Resample data to the specified timeframe."""
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, unit='ns', utc=True)
        
        resampled = df.groupby('symbol').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        return self.map_symbols(resampled)

    def save_processed_data(self, df, timeframe):
        """
        Save processed data to a parquet file with correct timestamps.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            timeframe (str): The timeframe of the data (e.g., '1D', '1H').

        Raises:
            Exception: If there's an error during the saving process.
        """
        file_path = os.path.join(self.processed_dir, f"processed_{timeframe}.parquet")
        
        try:
            logging.info(f"Saving processed data for timeframe {timeframe}. Shape: {df.shape}")
            logging.info(f"Index type before processing: {type(df.index)}")
            
            # Ensure 'timestamp' column exists and is in the correct format
            if 'timestamp' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = df.index
                else:
                    raise ValueError("DataFrame does not contain a 'timestamp' column or a DatetimeIndex")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Set 'timestamp' as the index
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"Index type after processing: {type(df.index)}")
            logging.info(f"Date range before saving: {df.index.min()} to {df.index.max()}")

            # Save the DataFrame to parquet format
            df.to_parquet(file_path, engine='pyarrow', index=True)

            # Verify the saved data
            saved_df = pd.read_parquet(file_path, engine='pyarrow')
            logging.info(f"Verified saved data. Shape: {saved_df.shape}")
            logging.info(f"Date range in saved file: {saved_df.index.min()} to {saved_df.index.max()}")

            # Save additional metadata
            txt_file = file_path.replace('.parquet', '.txt')
            with open(txt_file, 'w') as f:
                f.write(f"Date range: {df.index.min()} to {df.index.max()}\n")
                f.write(f"Shape: {df.shape}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")

            logging.info(f"Successfully saved processed data for timeframe {timeframe}")

        except Exception as e:
            logging.error(f"Error saving processed data for timeframe {timeframe}: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_data(self, symbols, start_date, end_date, timeframe='1T'):
        """
        Get data for specified symbols, date range, and timeframe.
        Attempts to load from processed data first, if not available, processes raw data.
        """
        processed_file = os.path.join(self.processed_dir, f"processed_{timeframe}.parquet")
        
        logging.info(f"Attempting to load processed data for timeframe {timeframe}")
        
        if os.path.exists(processed_file):
            try:
                df = pd.read_parquet(processed_file, engine='pyarrow')
                logging.info(f"Successfully read parquet file. Shape: {df.shape}")
                
                if df.empty:
                    logging.warning(f"Processed file for timeframe {timeframe} is empty")
                    return pd.DataFrame()
                
                # Ensure the index is a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                elif df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                start_date = pd.to_datetime(start_date, utc=True)
                end_date = pd.to_datetime(end_date, utc=True)
                
                logging.info(f"Date range in file: {df.index.min()} to {df.index.max()}")
                logging.info(f"Requested date range: {start_date} to {end_date}")
                
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if df.empty:
                    logging.warning(f"No data left after date filtering for timeframe {timeframe}")
                    return pd.DataFrame()
                
                if symbols != ["ALL_SYMBOLS"]:
                    df = df[df['symbol'].isin(symbols)]
                
                if df.empty:
                    logging.warning(f"No data left after symbol filtering for timeframe {timeframe}")
                    return pd.DataFrame()
                
                logging.info(f"Returning processed data for timeframe {timeframe}. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Error loading processed data: {str(e)}")
                logging.info("Falling back to processing raw data.")
        else:
            logging.warning(f"Processed file does not exist for timeframe {timeframe}")

        logging.info(f"Processing raw data for timeframe {timeframe}")
        
        # Process files and concatenate the results
        dfs = []
        for df in self.process_files(start_date, end_date, symbols):
            dfs.append(df)
        
        if not dfs:
            logging.warning("No data found for the specified criteria.")
            return pd.DataFrame()
        
        df = pd.concat(dfs)
        
        # Resample the data to the desired timeframe
        df = self.resample_data(df, timeframe)
        
        if not df.empty:
            self.save_processed_data(df, timeframe)
        else:
            logging.warning("No data available after processing.")
        
        return df

    def map_symbols(self, df):
        """Map symbols to their corresponding futures."""
        df['symbol'] = df['symbol'].map(self.symbol_map).fillna(df['symbol'])
        return df

    def print_error_summary(self):
        """
        Print a summary of files that couldn't be processed.
        """
        if self.error_files:
            print("\nFiles that couldn't be processed:")
            for file, error in self.error_files:
                print(f"{file}: {error}")
        else:
            print("\nAll files processed successfully.")