import os
import databento as db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from tqdm import tqdm
from src.backtester import Backtester
from src.visualizer import Visualizer
from src.data_loader import DataLoader
from src.divergence import calc_divergence
from src.indicators import add_indicators

def process_timeframe(loader, tf, symbols, start_date, end_date):
    """
    Process data for a specific timeframe with detailed error logging.

    Args:
        loader (DataLoader): Instance of DataLoader.
        tf (str): Timeframe to process.
        symbols (list): List of symbols to process.
        start_date (str): Start date for data processing.
        end_date (str): End date for data processing.

    Returns:
        tuple: Timeframe and processed DataFrame.
    """
    print(f"Processing timeframe: {tf}")
    tf_data = pd.DataFrame()
    
    try:
        for i, chunk in enumerate(tqdm(loader.get_data(symbols, start_date, end_date, timeframe=tf), desc=f"Processing {tf}")):
            try:
                print(f"Processing chunk {i+1} for timeframe {tf}")
                print(f"Chunk shape: {chunk.shape}")
                print(f"Chunk columns: {chunk.columns}")
                print(f"Chunk dtypes: {chunk.dtypes}")
                print(f"Chunk index type: {type(chunk.index)}")
                
                # Ensure index is DatetimeIndex
                if not isinstance(chunk.index, pd.DatetimeIndex):
                    chunk.index = pd.to_datetime(chunk.index)
                    print("Converted index to DatetimeIndex")
                
                # Add indicators
                chunk = add_indicators(chunk)
                print("Added indicators")
                
                # Calculate divergence
                chunk['div_original'] = calc_divergence(chunk, 'vwap', 14, True, 5)
                print("Calculated divergence")
                
                tf_data = pd.concat([tf_data, chunk], ignore_index=True)
                print(f"Concatenated chunk. Current tf_data shape: {tf_data.shape}")
            
            except Exception as e:
                print(f"Error processing chunk {i+1} in timeframe {tf}: {str(e)}")
                print(f"Chunk that caused error: {chunk}")
                continue
        
        print(f"Completed processing timeframe: {tf}")
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error processing timeframe {tf}: {str(e)} - {tb}")
    
    if tf_data.empty:
        print(f"Warning: No data processed for timeframe {tf}")
    else:
        print(f"Processed data shape for timeframe {tf}: {tf_data.shape}")
    
    return tf, tf_data

def main(verbose=False):
    loader = DataLoader(api_key=os.getenv("DATABENTO_API_KEY", "API_KEY_HERE"))

    symbols = ["ALL_SYMBOLS"]
    start_date = "2019-08-16"  # Update this to match the actual start date in your data files
    end_date = "2019-08-23"    # Update this to match the actual end date in your data files
    timeframes = ['1T', '3T', '5T', '15T', '1H', '4H', '12H', '1D', '2D']

    print("\nLoading and processing data...")
    all_data = {}
    
    for tf in tqdm(timeframes, desc="Overall Progress"):
        try:
            print(f"\nProcessing timeframe: {tf}")
            tf_data = pd.DataFrame()
            for chunk in loader.get_data(symbols, start_date, end_date, timeframe=tf):
                if not chunk.empty:
                    tf_data = pd.concat([tf_data, chunk], ignore_index=True)
            
            if not tf_data.empty:
                all_data[tf] = tf_data
                print(f"Successfully processed {tf} timeframe.")
                print(f"Shape: {tf_data.shape}")
                print(f"Columns: {tf_data.columns}")
                print(f"Data types: {tf_data.dtypes}")
                print(f"Date range: {tf_data.index.min()} to {tf_data.index.max()}")
                if verbose:
                    print("First few rows:")
                    print(tf_data.head())
            else:
                print(f"Warning: No data processed for {tf} timeframe")
                all_data[tf] = pd.DataFrame()
        
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error processing timeframe {tf}: {str(e)}")
            print(f"Traceback: {tb}")
            all_data[tf] = pd.DataFrame()

    print("\nData processing complete.")
    loader.print_error_summary()

    if all(df.empty for df in all_data.values()):
        print("No data could be processed. Exiting.")
        return

    if all_data['1T'].empty:
        print("1T data not available for backtesting. Skipping backtest.")
    else:
        print("\nRunning backtest...")
        try:
            backtester = Backtester(all_data['1T'])
            results = backtester.run(lambda x: x[x['div_original']])
            metrics = backtester.calculate_metrics()
            print("\nBacktest Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"Error during backtesting: {str(e)}")

    print("\nGenerating visualizations...")
    if not all_data['1T'].empty:
        try:
            visualizer = Visualizer(all_data['1T'])
            visualizer.plot_candlestick(symbols[0], start_date, end_date)
            visualizer.plot_indicator(symbols[0], 'vwap', start_date, end_date)
            visualizer.plot_divergence(symbols[0], 'div_original', start_date, end_date)
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
    else:
        print("Not enough data for visualization.")

if __name__ == "__main__":
    main(verbose=True)