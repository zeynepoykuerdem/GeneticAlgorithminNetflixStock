import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pandas as pd

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all available metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred)
    }

def create_fitness_function(train_data: np.ndarray, 
                          test_data: np.ndarray, 
                          model_class: Any,
                          metric: str = 'RMSE') -> callable:
    """
    Create a fitness function for the genetic algorithm.
    
    Args:
        train_data: Training time series data
        test_data: Test time series data for validation
        model_class: Forecasting model class
        metric: Metric to optimize ('MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE')
    
    Returns:
        Fitness function that takes parameters and returns fitness score
    """
    def fitness_function(parameters: np.ndarray) -> float:
        try:
            # Create and fit model
            model = model_class()
            model.fit(train_data, parameters)
            
            # Make predictions
            predictions = model.predict(len(test_data))
            
            # Calculate fitness (negative error for maximization)
            if metric == 'MAE':
                error = mean_absolute_error(test_data, predictions)
            elif metric == 'MSE':
                error = mean_squared_error(test_data, predictions)
            elif metric == 'RMSE':
                error = root_mean_squared_error(test_data, predictions)
            elif metric == 'MAPE':
                error = mean_absolute_percentage_error(test_data, predictions)
            elif metric == 'SMAPE':
                error = symmetric_mean_absolute_percentage_error(test_data, predictions)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Return negative error as fitness (higher is better)
            return -error if not np.isnan(error) and not np.isinf(error) else -1e6
            
        except Exception as e:
            # Return very low fitness for invalid parameters
            return -1e6
    
    return fitness_function

def plot_time_series(data: np.ndarray, 
                    predictions: np.ndarray = None,
                    train_size: int = None,
                    title: str = "Time Series Forecast",
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot time series data with optional predictions.
    
    Args:
        data: Original time series data
        predictions: Forecasted values
        train_size: Size of training data (for split visualization)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data
    if train_size is not None:
        ax.plot(range(train_size), data[:train_size], 
               label='Training Data', color='blue', alpha=0.7)
        ax.plot(range(train_size, len(data)), data[train_size:], 
               label='Test Data', color='green', alpha=0.7)
    else:
        ax.plot(range(len(data)), data, label='Original Data', color='blue', alpha=0.7)
    
    # Plot predictions
    if predictions is not None:
        if train_size is not None:
            pred_x = range(train_size, train_size + len(predictions))
        else:
            pred_x = range(len(data), len(data) + len(predictions))
        ax.plot(pred_x, predictions, label='Predictions', 
               color='red', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_ga_evolution(fitness_history: List[Dict], 
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot genetic algorithm evolution over generations.
    
    Args:
        fitness_history: List of fitness statistics per generation
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    generations = range(1, len(fitness_history) + 1)
    best_fitness = [gen['best'] for gen in fitness_history]
    avg_fitness = [gen['average'] for gen in fitness_history]
    worst_fitness = [gen['worst'] for gen in fitness_history]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(generations, best_fitness, label='Best Fitness', color='green', linewidth=2)
    ax.plot(generations, avg_fitness, label='Average Fitness', color='blue', linewidth=1)
    ax.plot(generations, worst_fitness, label='Worst Fitness', color='red', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Genetic Algorithm Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_models_performance(results: Dict[str, Dict[str, Any]], 
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Compare performance of different models.
    
    Args:
        results: Dictionary with model names as keys and results as values
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    metrics = ['MAE', 'RMSE', 'MAPE']
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        values = [results[model]['metrics'][metric] for model in models if metric in results[model]['metrics']]
        valid_models = [model for model in models if metric in results[model]['metrics']]
        
        bars = axes[idx].bar(valid_models, values, alpha=0.7)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_ylabel(metric)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_results_summary(model_name: str,
                          parameters: np.ndarray,
                          metrics: Dict[str, float],
                          ga_generations: int,
                          execution_time: float) -> str:
    """
    Create a formatted summary of model results.
    
    Args:
        model_name: Name of the forecasting model
        parameters: Optimized parameters
        metrics: Performance metrics
        ga_generations: Number of GA generations
        execution_time: Total execution time
    
    Returns:
        Formatted summary string
    """
    summary = f"""
{'='*60}
MODEL PERFORMANCE SUMMARY
{'='*60}
Model: {model_name}
GA Generations: {ga_generations}
Execution Time: {execution_time:.2f} seconds

Optimized Parameters:
{', '.join([f'{param:.4f}' for param in parameters])}

Performance Metrics:
"""
    
    for metric, value in metrics.items():
        summary += f"  {metric:8s}: {value:10.4f}\n"
    
    summary += f"{'='*60}\n"
    
    return summary