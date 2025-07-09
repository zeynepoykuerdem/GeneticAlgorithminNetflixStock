import numpy as np
import pandas as pd
from typing import Dict, Any

class TimeSeriesGenerator:
    """Generate synthetic time series data for testing forecasting models."""
    
    @staticmethod
    def generate_trend_series(length: int, 
                            trend_slope: float = 0.1, 
                            noise_level: float = 0.1,
                            start_value: float = 100) -> np.ndarray:
        """Generate time series with linear trend and noise."""
        t = np.arange(length)
        trend = start_value + trend_slope * t
        noise = np.random.normal(0, noise_level * np.std(trend), length)
        return trend + noise
    
    @staticmethod
    def generate_seasonal_series(length: int,
                               season_length: int = 12,
                               amplitude: float = 10,
                               trend_slope: float = 0.05,
                               noise_level: float = 0.1,
                               start_value: float = 100) -> np.ndarray:
        """Generate time series with seasonal pattern, trend, and noise."""
        t = np.arange(length)
        
        # Trend component
        trend = start_value + trend_slope * t
        
        # Seasonal component
        seasonal = amplitude * np.sin(2 * np.pi * t / season_length)
        
        # Noise component
        noise = np.random.normal(0, noise_level * amplitude, length)
        
        return trend + seasonal + noise
    
    @staticmethod
    def generate_ar_series(length: int,
                          ar_params: list = [0.7, -0.2],
                          noise_level: float = 1.0,
                          start_value: float = 100) -> np.ndarray:
        """Generate autoregressive time series."""
        series = np.zeros(length)
        ar_order = len(ar_params)
        
        # Initialize with random values
        for i in range(ar_order):
            series[i] = start_value + np.random.normal(0, noise_level)
        
        # Generate AR process
        for t in range(ar_order, length):
            ar_component = sum(ar_params[i] * series[t - i - 1] for i in range(ar_order))
            noise = np.random.normal(0, noise_level)
            series[t] = ar_component + noise
        
        return series
    
    @staticmethod
    def generate_complex_series(length: int,
                              trend_slope: float = 0.02,
                              season_length: int = 12,
                              seasonal_amplitude: float = 5,
                              ar_params: list = [0.5, -0.1],
                              noise_level: float = 0.5,
                              start_value: float = 100) -> np.ndarray:
        """Generate complex time series with trend, seasonality, and AR components."""
        t = np.arange(length)
        
        # Trend component
        trend = start_value + trend_slope * t
        
        # Seasonal component
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / season_length)
        
        # AR component
        ar_series = np.zeros(length)
        ar_order = len(ar_params)
        
        # Initialize AR series
        for i in range(min(ar_order, length)):
            ar_series[i] = np.random.normal(0, noise_level)
        
        # Generate AR process
        for t_ar in range(ar_order, length):
            ar_component = sum(ar_params[i] * ar_series[t_ar - i - 1] for i in range(ar_order))
            ar_series[t_ar] = ar_component + np.random.normal(0, noise_level)
        
        return trend + seasonal + ar_series
    
    @staticmethod
    def generate_financial_like_series(length: int,
                                     initial_price: float = 100,
                                     volatility: float = 0.02,
                                     drift: float = 0.0001) -> np.ndarray:
        """Generate financial price-like series using geometric Brownian motion."""
        dt = 1  # Time step
        prices = np.zeros(length)
        prices[0] = initial_price
        
        for t in range(1, length):
            random_shock = np.random.normal(0, 1)
            price_change = drift * prices[t-1] * dt + volatility * prices[t-1] * random_shock * np.sqrt(dt)
            prices[t] = prices[t-1] + price_change
        
        return prices
    
    @staticmethod
    def add_outliers(series: np.ndarray, 
                    outlier_probability: float = 0.05,
                    outlier_magnitude: float = 3.0) -> np.ndarray:
        """Add random outliers to a time series."""
        series_with_outliers = series.copy()
        std_dev = np.std(series)
        
        for i in range(len(series)):
            if np.random.random() < outlier_probability:
                outlier_direction = 1 if np.random.random() < 0.5 else -1
                series_with_outliers[i] += outlier_direction * outlier_magnitude * std_dev
        
        return series_with_outliers
    
    @classmethod
    def create_sample_datasets(cls) -> Dict[str, Dict[str, Any]]:
        """Create a collection of sample datasets for testing."""
        np.random.seed(42)  # For reproducibility
        
        datasets = {
            'simple_trend': {
                'data': cls.generate_trend_series(100, trend_slope=0.5, noise_level=0.2),
                'description': 'Simple linear trend with noise'
            },
            'seasonal': {
                'data': cls.generate_seasonal_series(120, season_length=12, amplitude=15, trend_slope=0.1),
                'description': 'Seasonal pattern with annual cycle and trend'
            },
            'autoregressive': {
                'data': cls.generate_ar_series(150, ar_params=[0.7, -0.2], noise_level=2.0),
                'description': 'Autoregressive process AR(2)'
            },
            'complex': {
                'data': cls.generate_complex_series(200, trend_slope=0.05, seasonal_amplitude=10),
                'description': 'Complex series with trend, seasonality, and AR components'
            },
            'financial': {
                'data': cls.generate_financial_like_series(180, initial_price=100, volatility=0.03),
                'description': 'Financial price-like series using geometric Brownian motion'
            }
        }
        
        # Add outliers to one dataset
        datasets['seasonal_with_outliers'] = {
            'data': cls.add_outliers(datasets['seasonal']['data'], outlier_probability=0.08),
            'description': 'Seasonal pattern with outliers'
        }
        
        return datasets