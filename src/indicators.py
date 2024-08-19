import pandas as pd
import numpy as np
import traceback

def calculate_ema(df, period):
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'timestamp', f'ema_{period}'])
    
    # Check if 'symbol' is in the index
    if 'symbol' in df.index.names:
        ema = df.groupby(level='symbol')['close'].ewm(span=period, adjust=False).mean().reset_index()
    else:
        ema = df.groupby('symbol')['close'].ewm(span=period, adjust=False).mean().reset_index()
    
    ema = ema.rename(columns={'close': f'ema_{period}'})
    ema['timestamp'] = df.index.get_level_values('timestamp') if 'timestamp' in df.index.names else df['timestamp']
    return ema

def calculate_vwap(df):
    if df.empty:
        return df
    df = df.copy()
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return df

def calculate_rsi(df, period=14):
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'timestamp', 'rsi'])
    
    df = df.reset_index()
    df = df.sort_values(['symbol', 'timestamp'])
    
    # Ensure 'symbol' is treated as a string
    df['symbol'] = df['symbol'].astype(str)
    
    delta = df.groupby('symbol')['close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby(df['symbol']).rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df['symbol']).rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return pd.DataFrame({'symbol': df['symbol'], 'timestamp': df['timestamp'], 'rsi': rsi})

def add_indicators(df, config):
    if df.empty:
        return df

    print("Adding indicators to DataFrame")
    print(f"DataFrame shape before adding indicators: {df.shape}")
    print(f"DataFrame columns before adding indicators: {df.columns}")
    print(f"Data types: {df.dtypes}")
    print(f"Unique symbols: {df['symbol'].unique()}")

    try:
        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame does not have a 'timestamp' column to set as index")

        df = calculate_vwap(df)
        
        # Calculate EMAs for different timeframes
        for timeframe in ['1T', '5T', '1H', '1D']:
            # Group by symbol before resampling to preserve the column
            resampled_df = df.groupby('symbol').resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            ema_df = calculate_ema(resampled_df, 27)
            # Merge based on both 'symbol' and 'timestamp'
            df = df.reset_index().merge(ema_df, on=['symbol', 'timestamp'], how='left', suffixes=('', f'_ema_{timeframe}')).set_index('timestamp')

        # Process RSI
        rsi_df = calculate_rsi(df)
        
        # Ensure both DataFrames have the same index type before merging
        df = df.reset_index()
        rsi_df = rsi_df.reset_index()
        df = pd.merge(df, rsi_df, on=['symbol', 'timestamp'], how='left')
        df = df.set_index('timestamp')

        print("Successfully added indicators")
        print(f"DataFrame shape after adding indicators: {df.shape}")
        print(f"DataFrame columns after adding indicators: {df.columns}")
    except Exception as e:
        print(f"Error in add_indicators: {str(e)}")
        print(traceback.format_exc())
        raise

    return df