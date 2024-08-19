import matplotlib.pyplot as plt
import mplfinance as mpf

class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot_candlestick(self, symbol, start_date, end_date):
        df = self.data[(self.data['symbol'] == symbol) & 
                       (self.data.index >= start_date) & 
                       (self.data.index <= end_date)]
        df = df.set_index('timestamp')
        mpf.plot(df, type='candle', style='charles', 
                 title=f'{symbol} Candlestick Chart',
                 ylabel='Price')

    def plot_indicator(self, symbol, indicator, start_date, end_date):
        df = self.data[(self.data['symbol'] == symbol) & 
                       (self.data.index >= start_date) & 
                       (self.data.index <= end_date)]
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[indicator])
        plt.title(f'{symbol} {indicator} Chart')
        plt.xlabel('Date')
        plt.ylabel(indicator)
        plt.show()

    def plot_divergence(self, symbol, divergence_series, start_date, end_date):
        df = self.data[(self.data['symbol'] == symbol) & 
                       (self.data.index >= start_date) & 
                       (self.data.index <= end_date)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['close'], label='Close Price')
        ax.scatter(df[divergence_series].index, df[divergence_series]['close'], 
                   color='red', label='Divergence')
        ax.set_title(f'{symbol} Divergence Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        plt.show()