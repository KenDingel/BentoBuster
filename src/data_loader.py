import os
import json
import databento as db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zstandard as zstd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from databento.common.error import BentoError

class DataLoader:
    """
    A class for loading and processing financial data from local files.

    This class handles data decompression, loading, and processing from .dbn and .dbn.zst files.
    It supports multi-core processing for improved performance.

    Attributes:
        client (db.Historical): Databento historical data client.
        data_dir (str): Directory containing the data files.
        num_cores (int): Number of CPU cores to use for parallel processing.
        error_files (list): List of files that couldn't be processed, with error messages.
        symbology (dict): Mapping of instrument IDs to symbols.
        metadata (dict): Metadata for the loaded data.
    """

    def __init__(self, api_key, data_dir='data'):
        """
        Initialize the DataLoader.

        Args:
            api_key (str): Databento API key.
            data_dir (str, optional): Directory containing the data files. Defaults to 'data'.
        """
        self.client = db.Historical(api_key)
        self.data_dir = data_dir
        self.num_cores = mp.cpu_count()
        self.error_files = []
        self.symbology = self.load_symbology()
        self.metadata = self.load_metadata()

    def load_symbology(self):
        """
        Load symbology mapping from a JSON file.

        Returns:
            dict: Mapping of instrument IDs to symbols.
        """
        symbology_file = os.path.join(self.data_dir, 'symbology.json')
        if os.path.exists(symbology_file):
            with open(symbology_file, 'r') as f:
                return json.load(f)
        return {}

    def load_metadata(self):
        """
        Load metadata from a JSON file.

        Returns:
            dict: Metadata for the loaded data.
        """
        metadata_file = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def decompress_file(self, file_path):
        """
        Decompress a .dbn.zst file to .dbn.

        Args:
            file_path (str): Path to the compressed file.

        Returns:
            str: Path to the decompressed file, or None if decompression failed.
        """
        output_file = file_path.replace('.dbn.zst', '.dbn')
        try:
            with open(file_path, 'rb') as compressed, open(output_file, 'wb') as decompressed:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(compressed, decompressed)
            return output_file
        except Exception as e:
            self.error_files.append((file_path, f"Decompression error: {str(e)}"))
            return None

    def process_file(self, filename):
        """
        Process a single data file.

        Args:
            filename (str): Name of the file to process.

        Returns:
            pd.DataFrame: Processed data, or an empty DataFrame if processing failed.
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            if filename.endswith('.dbn.zst'):
                file_path = self.decompress_file(file_path)
                if file_path is None:
                    return pd.DataFrame()
            
            data = db.DBNStore.from_file(file_path)
            df = data.to_df()
            
            if df.empty:
                self.error_files.append((filename, "Empty DataFrame"))
                return pd.DataFrame()
            
            if 'symbol' not in df.columns and 'instrument_id' in df.columns:
                df['symbol'] = df['instrument_id'].map(self.symbology.get)
            
            return df
        except BentoError as e:
            self.error_files.append((filename, f"BentoError: {str(e)}"))
            return pd.DataFrame()
        except Exception as e:
            self.error_files.append((filename, f"Unexpected error: {str(e)}"))
            return pd.DataFrame()

    def load_local_data(self, start_date, end_date, symbols, chunk_size=timedelta(days=1)):
        """
        Load and process local data in chunks.

        Args:
            start_date (str or datetime): Start date for data loading.
            end_date (str or datetime): End date for data loading.
            symbols (list): List of symbols to load data for.
            chunk_size (timedelta, optional): Size of each data chunk. Defaults to 1 day.

        Yields:
            pd.DataFrame: Processed data chunk.
        """
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)
        current_date = start_date

        while current_date <= end_date:
            chunk_end = min(current_date + chunk_size, end_date)
            files = [f for f in os.listdir(self.data_dir) if f.endswith(('.dbn.zst', '.dbn'))]
            
            print(f"Processing data from {current_date} to {chunk_end}")
            print(f"Found {len(files)} files to process")
            
            with mp.Pool(processes=self.num_cores) as pool:
                results = list(tqdm(pool.imap(self.process_file, files), total=len(files), desc="Processing files"))
            
            valid_results = [df for df in results if not df.empty]
            if valid_results:
                combined_df = pd.concat(valid_results, ignore_index=True)
                combined_df = combined_df[(combined_df.index >= current_date) & (combined_df.index < chunk_end)]
                if symbols != ["ALL_SYMBOLS"]:
                    combined_df = combined_df[combined_df['symbol'].isin(symbols)]
                if not combined_df.empty:
                    print(f"Yielding chunk with shape: {combined_df.shape}")
                    yield combined_df
                else:
                    print("No valid data in this chunk after filtering")
            else:
                print("No valid data in this chunk")
            
            current_date = chunk_end

    def get_data(self, symbols, start_date, end_date, timeframe='1T'):
        """
        Get data for specified symbols, date range, and timeframe.

        Args:
            symbols (list): List of symbols to get data for.
            start_date (str or datetime): Start date for data retrieval.
            end_date (str or datetime): End date for data retrieval.
            timeframe (str, optional): Timeframe for data resampling. Defaults to '1T' (1 minute).

        Yields:
            pd.DataFrame: Processed and resampled data chunk.
        """
        for chunk in self.load_local_data(start_date, end_date, symbols):
            if timeframe != '1T':
                print(f"Resampling data to {timeframe}")
                chunk = self.resample_data(chunk, timeframe)
            yield chunk

    def resample_data(self, df, timeframe):
        """
        Resample data to the specified timeframe.

        Args:
            df (pd.DataFrame): Data to resample.
            timeframe (str): Timeframe for resampling.

        Returns:
            pd.DataFrame: Resampled data.
        """
        resampled = df.groupby('symbol').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        return resampled

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