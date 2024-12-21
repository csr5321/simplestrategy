import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import os
from datetime import datetime, timedelta
import time
from enum import Enum
from typing import Dict, Optional, Union, List
import logging

class TimeFrame(Enum):
    INTRADAY_1MIN = '1min'
    INTRADAY_5MIN = '5min'
    INTRADAY_15MIN = '15min'
    INTRADAY_30MIN = '30min'
    INTRADAY_60MIN = '60min'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class HybridTradingModel:
    def __init__(self, api_key: str, lookback_period: int = 20, num_std: float = 2, 
                 hedge_ratio: float = 0.2, rate_limit_pause: float = 12.1):
        """
        Initialize the hybrid trading model with enhanced Alpha Vantage integration.
        
        Parameters:
        api_key (str): Alpha Vantage API key
        lookback_period (int): Period for calculating moving averages and bands
        num_std (float): Number of standard deviations for Bollinger Bands
        hedge_ratio (float): Proportion of position to hedge
        rate_limit_pause (float): Pause between API calls in seconds (Alpha Vantage limit: 5 calls per minute)
        """
        self.api_key = api_key
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.hedge_ratio = hedge_ratio
        self.rate_limit_pause = rate_limit_pause
        self.logger = logging.getLogger(__name__)
        
        # Initialize technical indicator cache
        self.indicator_cache = {}
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.hedge_ratio = hedge_ratio
        self.scaler = StandardScaler()
        self.models = {
            'linear': LinearRegression(),
            'quadratic': LinearRegression(),
            'cubic': LinearRegression()
        }
        self.weights = {'linear': 0.33, 'quadratic': 0.33, 'cubic': 0.34}
        
    def _rate_limit_pause(self):
        """Pause execution to comply with API rate limits."""
        time.sleep(self.rate_limit_pause)

    async def fetch_data(self, symbol: str, timeframe: TimeFrame, 
                        lookback_days: int = 1095) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage with specified timeframe.
        
        Parameters:
        symbol (str): Stock symbol
        timeframe (TimeFrame): Timeframe enum value
        lookback_days (int): Number of days of historical data to fetch
        
        Returns:
        Optional[pd.DataFrame]: Processed market data
        """
        try:
            if timeframe in [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]:
                func = getattr(self.ts, f'get_{timeframe.value}')
                data, _ = func(symbol=symbol, outputsize='full')
            else:
                data, _ = self.ts.get_intraday(symbol=symbol, 
                                             interval=timeframe.value,
                                             outputsize='full')
            
            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Filter for lookback period
            start_date = datetime.now() - timedelta(days=lookback_days)
            data = data[data.index > start_date]
            
            self._rate_limit_pause()
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching {timeframe.value} data for {symbol}: {str(e)}")
            return None

    async def fetch_technical_indicators(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch additional technical indicators from Alpha Vantage.
        
        Parameters:
        symbol (str): Stock symbol
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of technical indicators
        """
        indicators = {}
        try:
            # RSI
            rsi_data, _ = self.ti.get_rsi(symbol=symbol, interval='daily',
                                         time_period=14, series_type='close')
            indicators['RSI'] = rsi_data
            self._rate_limit_pause()

            # MACD
            macd_data, _ = self.ti.get_macd(symbol=symbol, interval='daily',
                                           series_type='close')
            indicators['MACD'] = macd_data
            self._rate_limit_pause()

            # ADX
            adx_data, _ = self.ti.get_adx(symbol=symbol, interval='daily',
                                         time_period=14)
            indicators['ADX'] = adx_data
            self._rate_limit_pause()

            # Stochastic Oscillator
            stoch_data, _ = self.ti.get_stoch(symbol=symbol, interval='daily')
            indicators['STOCH'] = stoch_data
            self._rate_limit_pause()

            return indicators

        except Exception as e:
            self.logger.error(f"Error fetching technical indicators for {symbol}: {str(e)}")
            return {}

    def preprocess_data(self, df: pd.DataFrame, 
                       technical_indicators: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Preprocess the input data and compute technical indicators.
        
        Parameters:
        df (pd.DataFrame): Raw market data
        technical_indicators (Dict[str, pd.DataFrame]): Additional technical indicators
        
        Returns:
        pd.DataFrame: Processed data with all indicators
        """
        df = df.copy()
        
        # Basic calculations
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.lookback_period).std()
        
        # Compute Bollinger Bands
        df['SMA'] = df['close'].rolling(window=self.lookback_period).mean()
        rolling_std = df['close'].rolling(window=self.lookback_period).std()
        df['upper_band'] = df['SMA'] + (rolling_std * self.num_std)
        df['lower_band'] = df['SMA'] - (rolling_std * self.num_std)
        
        # Add technical indicators if available
        if technical_indicators:
            for indicator_name, indicator_data in technical_indicators.items():
                # Align dates and add indicator columns
                common_dates = df.index.intersection(indicator_data.index)
                for col in indicator_data.columns:
                    df.loc[common_dates, f'{indicator_name}_{col}'] = indicator_data.loc[common_dates, col]
        
        return df.dropna()
    
    def generate_polynomial_features(self, X):
        """Generate polynomial features for regression models."""
        X_quad = np.square(X)
        X_cubic = np.power(X, 3)
        return {'linear': X, 'quadratic': X_quad, 'cubic': X_cubic}
    
    def fit_regression_models(self, df, prediction_window=5):
        """Fit linear, quadratic, and cubic regression models."""
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        poly_features = self.generate_polynomial_features(X)
        predictions = {}
        
        for model_type, model in self.models.items():
            X_poly = poly_features[model_type]
            X_train_poly = X_poly[:len(X_train)]
            X_val_poly = X_poly[len(X_train):]
            
            model.fit(X_train_poly, y_train)
            pred = model.predict(X_val_poly)
            mse = mean_squared_error(y_val, pred)
            predictions[model_type] = pred
            
            # Update model weights based on prediction error
            self.weights[model_type] = 1 / mse
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total_weight
            
        return predictions
    
    def calculate_position_sizes(self, df):
        """Calculate position sizes based on market conditions."""
        positions = pd.DataFrame(index=df.index)
        
        # Base position signals from Bollinger Bands
        positions['bb_signal'] = 0
        positions.loc[df['close'] < df['lower_band'], 'bb_signal'] = 1  # Buy signal
        positions.loc[df['close'] > df['upper_band'], 'bb_signal'] = -1  # Sell signal
        
        # Adjust position sizes based on volatility
        vol_percentile = df['volatility'].rank(pct=True)
        positions['volatility_adjustment'] = vol_percentile.apply(
            lambda x: 1.5 if x > 0.8 else 0.5 if x < 0.2 else 1.0
        )
        
        # Calculate final position sizes
        positions['position_size'] = positions['bb_signal'] * positions['volatility_adjustment']
        
        return positions
    
    def implement_risk_management(self, df, positions, max_loss=0.02):
        """Implement risk management rules."""
        risk_adjusted_positions = positions.copy()
        
        # Calculate stop-loss levels
        rolling_low = df['low'].rolling(window=self.lookback_period).min()
        stop_loss = df['close'] * (1 - max_loss)
        
        # Adjust positions based on stop-loss
        risk_adjusted_positions.loc[df['close'] < stop_loss, 'position_size'] = 0
        
        # Apply hedge ratio to reduce exposure
        risk_adjusted_positions['hedged_position'] = (
            risk_adjusted_positions['position_size'] * (1 - self.hedge_ratio)
        )
        
        return risk_adjusted_positions
    
    def calculate_performance_metrics(self, df, positions):
        """Calculate key performance metrics."""
        # Calculate returns
        strategy_returns = df['returns'] * positions['hedged_position'].shift(1)
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate drawdowns
        peak = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - peak) / peak
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
        
        metrics = {
            'cumulative_return': cumulative_returns.iloc[-1] - 1,
            'max_drawdown': drawdowns.min(),
            'sharpe_ratio': sharpe_ratio,
            'volatility': strategy_returns.std() * np.sqrt(252)
        }
        
        return metrics, cumulative_returns
    
    def plot_results(self, df, positions, cumulative_returns):
        """Plot strategy results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price and Bollinger Bands
        ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
        ax1.plot(df.index, df['upper_band'], 'r--', label='Upper Band', alpha=0.5)
        ax1.plot(df.index, df['lower_band'], 'g--', label='Lower Band', alpha=0.5)
        ax1.plot(df.index, df['SMA'], 'b--', label='SMA', alpha=0.5)
        
        # Plot positions
        ax1.scatter(df.index[positions['hedged_position'] > 0],
                   df['close'][positions['hedged_position'] > 0],
                   marker='^', color='g', label='Long')
        ax1.scatter(df.index[positions['hedged_position'] < 0],
                   df['close'][positions['hedged_position'] < 0],
                   marker='v', color='r', label='Short')
        
        ax1.set_title('Price Action and Trading Signals')
        ax1.legend()
        
        # Plot cumulative returns
        ax2.plot(df.index, cumulative_returns, label='Cumulative Returns')
        ax2.set_title('Strategy Cumulative Returns')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_strategy(self, df):
        """Execute the complete trading strategy."""
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Fit regression models
        predictions = self.fit_regression_models(processed_df)
        
        # Calculate positions
        positions = self.calculate_position_sizes(processed_df)
        
        # Apply risk management
        risk_adjusted_positions = self.implement_risk_management(processed_df, positions)
        
        # Calculate performance metrics
        metrics, cumulative_returns = self.calculate_performance_metrics(
            processed_df, risk_adjusted_positions
        )
        
        # Plot results
        self.plot_results(processed_df, risk_adjusted_positions, cumulative_returns)
        
        return metrics, risk_adjusted_positions, cumulative_returns

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Set your Alpha Vantage API key
        api_key = 'YOUR_API_KEY'  # Replace with your actual API key
        
        # Initialize the model
        model = HybridTradingModel(
            api_key=api_key,
            lookback_period=20,
            num_std=2,
            hedge_ratio=0.2
        )
        
        # Example: Run strategy on Apple stock with multiple timeframes
        symbol = 'COIN'
        timeframes = [
            TimeFrame.DAILY,
            # TimeFrame.INTRADAY_60MIN
        ]
        
        for timeframe in timeframes:
            # Fetch market data
            market_data = await model.fetch_data(symbol, timeframe)
            
            if market_data is not None:
                # Fetch technical indicators (for daily timeframe only)
                technical_indicators = {}
                if timeframe == TimeFrame.DAILY:
                    technical_indicators = await model.fetch_technical_indicators(symbol)
                
                # Process data and run strategy
                processed_data = model.preprocess_data(market_data, technical_indicators)
                metrics, positions, returns = model.run_strategy(processed_data)
                
                print(f"\nPerformance Metrics for {symbol} ({timeframe.value}):")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"Could not run strategy for {timeframe.value} due to data fetching error")
    
    # Run the async main function
    asyncio.run(main())