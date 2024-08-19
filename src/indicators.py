import pandas as pd
import numpy as np

def calculate_ema(df, period):
    if df.empty:
        return pd.DataFrame(columns=['symbol', f'ema_{period}'])
    if 'symbol' not in df.columns:
        print("Warning: 'symbol' column not found. Using 'instrument_id' as symbol.")
        df['symbol'] = df['instrument_id'].astype(str)
    
    print(f"Unique symbols: {df['symbol'].unique()}")
    print(f"Number of rows: {len(df)}")
    
    ema = df.groupby('symbol')['close'].ewm(span=period, adjust=False).mean().reset_index()
    ema = ema.rename(columns={'close': f'ema_{period}'})
    return ema

def calculate_vwap(df):
    if df.empty:
        return df
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return df

def calculate_rsi(df, period=14):
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'rsi'])
    if 'symbol' not in df.columns:
        print("Warning: 'symbol' column not found. Using 'instrument_id' as symbol.")
        df['symbol'] = df['instrument_id'].astype(str)
    
    delta = df.groupby('symbol')['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby('symbol').rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby('symbol').rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return pd.DataFrame({'symbol': df['symbol'], 'rsi': rsi}).reset_index()

def add_indicators(df):
    if df.empty:
        return df
    
    print("DataFrame columns before adding indicators:", df.columns)
    print("DataFrame info:")
    print(df.info())
    print("First few rows of the DataFrame:")
    print(df.head())
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    try:
        df = calculate_vwap(df)
        ema_df = calculate_ema(df, 27)
        df = df.merge(ema_df, on=['symbol', df.index.name], how='left')
        rsi_df = calculate_rsi(df)
        df = df.merge(rsi_df, on=['symbol', df.index.name], how='left')
    except Exception as e:
        print(f"Error in add_indicators: {str(e)}")
        print("DataFrame after error:")
        print(df.info())
        print(df.head())
        raise
    
    print("DataFrame columns after adding indicators:", df.columns)
    return df