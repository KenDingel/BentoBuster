import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from tqdm import tqdm
import traceback
import argparse
import configparser
from src.backtester import Backtester
from src.visualizer import Visualizer
from src.data_loader import DataLoader
from src.divergence import calc_divergence, calc_divergence_with_vwap, calc_divergence_multi_timeframe, calc_divergence_with_momentum, calc_divergence_with_confirmation
from src.indicators import add_indicators, calculate_ema, calculate_vwap, calculate_rsi

def parse_arguments():
    parser = argparse.ArgumentParser(description="BentoBuster: Technical Analysis Tool")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def process_chunk(chunk, timeframe, config):
    try:
        # Ensure 'timestamp' is the index
        if 'timestamp' in chunk.columns:
            chunk = chunk.set_index('timestamp')
        elif not isinstance(chunk.index, pd.DatetimeIndex):
            raise ValueError("Chunk does not have a 'timestamp' column or DatetimeIndex")
        
        # Ensure 'symbol' is a column and is of type string
        if 'symbol' not in chunk.columns:
            print(f"Warning: 'symbol' column not found in chunk for {timeframe} timeframe")
            print(f"Chunk columns: {chunk.columns}")
            return None
        chunk['symbol'] = chunk['symbol'].astype(str)
        
        print(f"Chunk data types before processing: {chunk.dtypes}")
        print(f"Unique symbols in chunk: {chunk['symbol'].unique()}")
        
        # Add indicators including divergence
        chunk = add_indicators(chunk, config)
        
        # Calculate divergence
        chunk['div_original'] = calc_divergence(chunk, 'vwap', 14, True, 5)
        chunk['div_vwap'] = calc_divergence_with_vwap(chunk, 'vwap', 14, True, 5, 20)
        chunk['div_multi_tf'] = calc_divergence_multi_timeframe(chunk, 'vwap', 14, True, 5, chunk['ema_1T'], chunk['ema_5T'], chunk['ema_1H'])
        chunk['div_momentum'] = calc_divergence_with_momentum(chunk, 'vwap', 14, True, 5, 10)
        chunk['div_confirmation'] = calc_divergence_with_confirmation(chunk, 'vwap', 14, True, 5, 3)
        
        print(f"Chunk data types after processing: {chunk.dtypes}")
        
        return chunk
    except Exception as e:
        print(f"Error processing chunk for {timeframe} timeframe: {str(e)}")
        print(traceback.format_exc())
        return None

def main(config, verbose=True):
    loader = DataLoader(data_dir='data')

    symbols = ["ALL_SYMBOLS"]
    start_date = "2023-088-27"
    end_date = "2024-08-16"
    timeframes = ['1T', '3T', '5T', '15T', '1H', '4H', '12H', '1D', '2D']

    print("\nStarting data processing with the following parameters:")
    print(f"Symbols: {symbols}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Timeframes: {timeframes}")

    all_data = {}
    
    for tf in tqdm(timeframes, desc="Overall Progress"):
        try:
            print(f"\nProcessing timeframe: {tf}")
            tf_data = loader.get_data(symbols, start_date, end_date, timeframe=tf)
            
            if not tf_data.empty:
                # Ensure 'symbol' is a column, not in the index
                if 'symbol' in tf_data.index.names:
                    tf_data = tf_data.reset_index()
                
                processed_chunks = []
                chunk_size = 100000  # Adjust this value based on your available memory
                
                for chunk_start in range(0, len(tf_data), chunk_size):
                    chunk = tf_data.iloc[chunk_start:chunk_start+chunk_size]
                    processed_chunk = process_chunk(chunk, tf, config)
                    if processed_chunk is not None:
                        processed_chunks.append(processed_chunk)
                
                if processed_chunks:
                    tf_data = pd.concat(processed_chunks)
                    
                    # Calculate divergences with settings 2222
                    tf_data['div_2222_1mv'] = calc_divergence(tf_data, 'vwap', 2, True, 2)
                    tf_data['div_2222_5mv'] = calc_divergence(tf_data, 'vwap', 2, True, 2)
                    
                    # Calculate divergences with settings 1212
                    tf_data['div_1212_1mv'] = calc_divergence(tf_data, 'vwap', 1, True, 2)
                    tf_data['div_1212_5mv'] = calc_divergence(tf_data, 'vwap', 2, True, 1)
                    
                    all_data[tf] = tf_data
                    print(f"Successfully processed {tf} timeframe.")
                    print(f"Shape: {tf_data.shape}")
                    print(f"Columns: {tf_data.columns}")
                    print(f"Data types: {tf_data.dtypes}")
                    print(f"Index type: {type(tf_data.index)}")
                    print(f"Date range: {tf_data.index.min()} to {tf_data.index.max()}")
                    print(f"Unique symbols: {tf_data['symbol'].unique()}")
                    if verbose:
                        print("First few rows:")
                        print(tf_data.head())
                else:
                    print(f"No data processed for {tf} timeframe")
            else:
                print(f"No data processed for {tf} timeframe")
        
        except Exception as e:
            print(f"Error processing timeframe {tf}: {str(e)}")
            print(traceback.format_exc())

    print("\nData processing complete.")
    loader.print_error_summary()

    if all(df.empty for df in all_data.values()):
        print("No data could be processed. Exiting.")
        return

    if '1T' not in all_data or all_data['1T'].empty:
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
            print(traceback.format_exc())

    print("\nGenerating visualizations...")
    if '1T' not in all_data or all_data['1T'].empty:
        print("Not enough data for visualization.")
    else:
        try:
            visualizer = Visualizer(all_data['1T'])
            symbol = all_data['1T']['symbol'].iloc[0]  # Use the first available symbol
            visualizer.plot_candlestick(symbol, start_date, end_date)
            visualizer.plot_indicator(symbol, 'vwap', start_date, end_date)
            visualizer.plot_divergence(symbol, 'div_original', start_date, end_date)
            # Add more visualizations as needed
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            print(traceback.format_exc())

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    main(config, verbose=args.verbose)