import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.optimize import minimize_scalar

class ForecastingModel(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def fit(self, data: np.ndarray, parameters: np.ndarray):
        """Fit the model to the data with given parameters."""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values for the specified number of steps."""
        pass
    
    @abstractmethod
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds for model parameters."""
        pass

class ExponentialSmoothingModel(ForecastingModel):
    """Exponential Smoothing forecasting model."""
    
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.level = None
        self.trend = None
        self.seasonal = None
        self.data = None
        self.season_length = 12  # Default seasonal period
        
    def fit(self, data: np.ndarray, parameters: np.ndarray):
        """
        Fit exponential smoothing model.
        Parameters: [alpha, beta, gamma] for level, trend, seasonal smoothing
        """
        self.data = data
        self.alpha, self.beta, self.gamma = parameters
        
        n = len(data)
        
        # Initialize level, trend, and seasonal components
        self.level = np.zeros(n)
        self.trend = np.zeros(n)
        self.seasonal = np.zeros(n)
        
        # Initial values
        self.level[0] = data[0]
        self.trend[0] = (data[1] - data[0]) if n > 1 else 0
        
        # Initialize seasonal component (simple average approach)
        if n >= self.season_length:
            for i in range(self.season_length):
                seasonal_values = []
                for j in range(i, n, self.season_length):
                    seasonal_values.append(data[j])
                self.seasonal[i] = np.mean(seasonal_values) - np.mean(data)
        
        # Update components using exponential smoothing
        for t in range(1, n):
            if t < self.season_length:
                seasonal_component = self.seasonal[t] if t < len(self.seasonal) else 0
            else:
                seasonal_component = self.seasonal[t - self.season_length]
            
            # Update level
            self.level[t] = self.alpha * (data[t] - seasonal_component) + \
                           (1 - self.alpha) * (self.level[t-1] + self.trend[t-1])
            
            # Update trend
            self.trend[t] = self.beta * (self.level[t] - self.level[t-1]) + \
                           (1 - self.beta) * self.trend[t-1]
            
            # Update seasonal component
            if t >= self.season_length:
                self.seasonal[t % self.season_length] = \
                    self.gamma * (data[t] - self.level[t]) + \
                    (1 - self.gamma) * seasonal_component
    
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values."""
        if self.level is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = np.zeros(steps)
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        
        for h in range(steps):
            # Level + trend component
            forecast = last_level + (h + 1) * last_trend
            
            # Add seasonal component if available
            if len(self.seasonal) >= self.season_length:
                seasonal_idx = (len(self.data) + h) % self.season_length
                forecast += self.seasonal[seasonal_idx]
            
            predictions[h] = forecast
        
        return predictions
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for alpha, beta, gamma parameters."""
        return [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

class PolynomialTrendModel(ForecastingModel):
    """Polynomial trend forecasting model."""
    
    def __init__(self):
        self.coefficients = None
        self.degree = None
        self.data_length = None
        
    def fit(self, data: np.ndarray, parameters: np.ndarray):
        """
        Fit polynomial model.
        Parameters: polynomial coefficients [a0, a1, a2, ...]
        """
        self.degree = len(parameters) - 1
        self.coefficients = parameters
        self.data_length = len(data)
        
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values using polynomial trend."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = np.zeros(steps)
        
        for h in range(steps):
            t = self.data_length + h + 1
            prediction = 0
            for i, coef in enumerate(self.coefficients):
                prediction += coef * (t ** i)
            predictions[h] = prediction
        
        return predictions
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for polynomial coefficients."""
        # Allow flexibility in coefficient values
        return [(-100, 100), (-10, 10), (-1, 1), (-0.1, 0.1)]

class ARIMASimpleModel(ForecastingModel):
    """Simplified ARIMA-like model (AR + MA components)."""
    
    def __init__(self, ar_order: int = 2, ma_order: int = 2):
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.ar_params = None
        self.ma_params = None
        self.residuals = None
        self.data = None
        
    def fit(self, data: np.ndarray, parameters: np.ndarray):
        """
        Fit ARIMA model.
        Parameters: [ar_params..., ma_params...]
        """
        self.data = data
        n_ar = self.ar_order
        n_ma = self.ma_order
        
        self.ar_params = parameters[:n_ar]
        self.ma_params = parameters[n_ar:n_ar + n_ma]
        
        # Calculate residuals for MA component
        self.residuals = np.zeros(len(data))
        fitted_values = np.zeros(len(data))
        
        for t in range(max(self.ar_order, self.ma_order), len(data)):
            # AR component
            ar_component = 0
            for i in range(self.ar_order):
                if t - i - 1 >= 0:
                    ar_component += self.ar_params[i] * data[t - i - 1]
            
            # MA component
            ma_component = 0
            for i in range(self.ma_order):
                if t - i - 1 >= 0:
                    ma_component += self.ma_params[i] * self.residuals[t - i - 1]
            
            fitted_values[t] = ar_component + ma_component
            self.residuals[t] = data[t] - fitted_values[t]
    
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values."""
        if self.ar_params is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = np.zeros(steps)
        extended_data = np.concatenate([self.data, predictions])
        extended_residuals = np.concatenate([self.residuals, np.zeros(steps)])
        
        for h in range(steps):
            t = len(self.data) + h
            
            # AR component
            ar_component = 0
            for i in range(self.ar_order):
                if t - i - 1 >= 0:
                    ar_component += self.ar_params[i] * extended_data[t - i - 1]
            
            # MA component (using zero future errors)
            ma_component = 0
            for i in range(self.ma_order):
                if t - i - 1 >= 0 and t - i - 1 < len(extended_residuals):
                    ma_component += self.ma_params[i] * extended_residuals[t - i - 1]
            
            predictions[h] = ar_component + ma_component
            extended_data[t] = predictions[h]
        
        return predictions
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for AR and MA parameters."""
        bounds = []
        # AR parameters
        for _ in range(self.ar_order):
            bounds.append((-0.99, 0.99))
        # MA parameters  
        for _ in range(self.ma_order):
            bounds.append((-0.99, 0.99))
        return bounds