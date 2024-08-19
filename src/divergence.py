import pandas as pd
import numpy as np
import traceback

def calc_divergence(df, velo, range_input, is_long, div_candles):
    """
    Calculate divergence based on various conditions with improved error handling.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and indicator data.
        velo (str): Column name for the velocity indicator.
        range_input (int): Range for rolling calculations.
        is_long (bool): If True, calculate bullish divergence; if False, bearish divergence.
        div_candles (int): Number of candles to look back for divergence calculation.

    Returns:
        pd.Series: Boolean series indicating divergence conditions.
    """
    if velo not in df.columns:
        raise KeyError(f"Column '{velo}' not found in DataFrame. Available columns: {df.columns}")
    
    df = df.sort_index()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Calculate conditions
    try:
        # Use .diff() instead of comparison with shifted values
        condition1 = df[velo].diff() > 0 if is_long else df[velo].diff() < 0
        condition2 = df['close'].diff() < 0 if is_long else df['close'].diff() > 0
        
        # Use .rolling().apply() for more complex conditions
        condition3 = df[velo].rolling(range_input).apply(lambda x: x.iloc[-1] > x.iloc[0] if is_long else x.iloc[-1] < x.iloc[0])
        condition4 = df['close'].rolling(range_input).apply(lambda x: x.iloc[-1] < x.iloc[0] if is_long else x.iloc[-1] > x.iloc[0])
        
        # Combine conditions
        divergence = condition1 & condition2 & condition3 & condition4
        
        # Fill NaN values with False
        divergence = divergence.fillna(False)
        
        return divergence
    
    except Exception as e:
        print(f"Error in calc_divergence: {str(e)}")
        print(traceback.format_exc())
        return pd.Series([False] * len(df), index=df.index)

def calc_divergence_with_vwap(df, velo, range_input, is_long, div_candles, vwap_range):
    """
    Calculate divergence with VWAP condition.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and indicator data.
        velo (str): Column name for the velocity indicator.
        range_input (int): Range for rolling calculations.
        is_long (bool): If True, calculate bullish divergence; if False, bearish divergence.
        div_candles (int): Number of candles to look back for divergence calculation.
        vwap_range (int): Range for VWAP calculation.

    Returns:
        pd.Series: Boolean series indicating divergence conditions with VWAP.
    """
    basic_div = calc_divergence(df, velo, range_input, is_long, div_candles)
    vwap_condition = (df['close'] > df['vwap']) & (df['close'].shift(div_candles) < df['vwap'].shift(div_candles)) if is_long else \
                     (df['close'] < df['vwap']) & (df['close'].shift(div_candles) > df['vwap'].shift(div_candles))
    return basic_div & vwap_condition

def calc_divergence_multi_timeframe(df, velo, range_input, is_long, div_candles, ema1, ema5, ema60):
    """
    Calculate divergence with multi-timeframe EMAs.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and indicator data.
        velo (str): Column name for the velocity indicator.
        range_input (int): Range for rolling calculations.
        is_long (bool): If True, calculate bullish divergence; if False, bearish divergence.
        div_candles (int): Number of candles to look back for divergence calculation.
        ema1 (pd.Series): 1-period EMA.
        ema5 (pd.Series): 5-period EMA.
        ema60 (pd.Series): 60-period EMA.

    Returns:
        pd.Series: Boolean series indicating divergence conditions with multi-timeframe EMAs.
    """
    basic_div = calc_divergence(df, velo, range_input, is_long, div_candles)
    ema_condition = (ema1 > ema5) & (ema5 > ema60) if is_long else (ema1 < ema5) & (ema5 < ema60)
    return basic_div & ema_condition

def calc_divergence_with_momentum(df, velo, range_input, is_long, div_candles, momentum_length):
    """
    Calculate divergence with momentum condition.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and indicator data.
        velo (str): Column name for the velocity indicator.
        range_input (int): Range for rolling calculations.
        is_long (bool): If True, calculate bullish divergence; if False, bearish divergence.
        div_candles (int): Number of candles to look back for divergence calculation.
        momentum_length (int): Length for momentum calculation.

    Returns:
        pd.Series: Boolean series indicating divergence conditions with momentum.
    """
    basic_div = calc_divergence(df, velo, range_input, is_long, div_candles)
    momentum = df['close'].diff(momentum_length)
    momentum_condition = momentum > 0 if is_long else momentum < 0
    return basic_div & momentum_condition

def calc_divergence_with_confirmation(df, velo, range_input, is_long, div_candles, confirm_candles):
    """
    Calculate divergence with confirmation condition.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and indicator data.
        velo (str): Column name for the velocity indicator.
        range_input (int): Range for rolling calculations.
        is_long (bool): If True, calculate bullish divergence; if False, bearish divergence.
        div_candles (int): Number of candles to look back for divergence calculation.
        confirm_candles (int): Number of candles for confirmation.

    Returns:
        pd.Series: Boolean series indicating divergence conditions with confirmation.
    """
    basic_div = calc_divergence(df, velo, range_input, is_long, div_candles)
    confirm_condition = df['close'].rolling(confirm_candles).apply(lambda x: x.is_monotonic_decreasing) if is_long else \
                        df['close'].rolling(confirm_candles).apply(lambda x: x.is_monotonic_increasing)
    return basic_div & confirm_condition