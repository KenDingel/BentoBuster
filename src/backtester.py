import pandas as pd

class Backtester:
    def __init__(self, data):
        self.data = data
        self.results = pd.DataFrame()

    def run(self, strategy_func, **kwargs):
        self.results = strategy_func(self.data, **kwargs)
        return self.results

    def calculate_returns(self):
        self.results['returns'] = self.results.groupby('symbol')['close'].pct_change()
        self.results['cumulative_returns'] = (1 + self.results['returns']).cumprod() - 1
        return self.results

    def calculate_metrics(self):
        returns = self.results['returns']
        metrics = {
            'Total Return': self.results['cumulative_returns'].iloc[-1],
            'Annualized Return': returns.mean() * 252,
            'Annualized Volatility': returns.std() * (252 ** 0.5),
            'Sharpe Ratio': (returns.mean() / returns.std()) * (252 ** 0.5),
            'Max Drawdown': (self.results['cumulative_returns'] / self.results['cumulative_returns'].cummax() - 1).min()
        }
        return pd.Series(metrics)