#!/usr/bin/env python3
"""
Time Series Forecasting with Genetic Algorithm Optimization

This program demonstrates the use of genetic algorithms to optimize
parameters for various time series forecasting models.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from genetic_algorithm import GeneticAlgorithm
from forecasting_models import ExponentialSmoothingModel, PolynomialTrendModel, ARIMASimpleModel
from data_generator import TimeSeriesGenerator
from utils import (create_fitness_function, plot_time_series, plot_ga_evolution, 
                  compare_models_performance, create_results_summary, calculate_all_metrics)

def split_data(data: np.ndarray, train_ratio: float = 0.8):
    """Split data into training and testing sets."""
    split_point = int(len(data) * train_ratio)
    return data[:split_point], data[split_point:]

def optimize_model(model_class, train_data, test_data, model_name, 
                  ga_params=None, metric='RMSE'):
    """Optimize a single model using genetic algorithm."""
    
    print(f"\nOptimizing {model_name}...")
    print(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    
    # Default GA parameters
    if ga_params is None:
        ga_params = {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
    
    # Create model instance to get parameter bounds
    model_instance = model_class()
    bounds = model_instance.get_parameter_bounds()
    
    # Create fitness function
    fitness_function = create_fitness_function(train_data, test_data, model_class, metric)
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        gene_bounds=bounds,
        population_size=ga_params['population_size'],
        mutation_rate=ga_params['mutation_rate'],
        crossover_rate=ga_params['crossover_rate']
    )
    
    # Run optimization
    start_time = time.time()
    best_individual = ga.run(fitness_function, ga_params['generations'], verbose=True)
    execution_time = time.time() - start_time
    
    # Evaluate best model
    final_model = model_class()
    final_model.fit(train_data, best_individual.genes)
    predictions = final_model.predict(len(test_data))
    metrics = calculate_all_metrics(test_data, predictions)
    
    return {
        'model': final_model,
        'parameters': best_individual.genes,
        'predictions': predictions,
        'metrics': metrics,
        'fitness_history': ga.fitness_history,
        'execution_time': execution_time,
        'ga_generations': ga_params['generations']
    }

def main():
    """Main execution function."""
    
    print("="*60)
    print("TIME SERIES FORECASTING WITH GENETIC ALGORITHMS")
    print("="*60)
    
    # Generate sample datasets
    print("\n1. Generating sample datasets...")
    datasets = TimeSeriesGenerator.create_sample_datasets()
    
    # Display available datasets
    print("\nAvailable datasets:")
    for name, info in datasets.items():
        print(f"  - {name}: {info['description']}")
    
    # Select a dataset for demonstration
    dataset_name = 'seasonal'  # Can be changed to test different datasets
    data = datasets[dataset_name]['data']
    
    print(f"\nSelected dataset: {dataset_name}")
    print(f"Dataset description: {datasets[dataset_name]['description']}")
    print(f"Dataset length: {len(data)}")
    
    # Split data
    train_data, test_data = split_data(data, train_ratio=0.8)
    
    # Define models to test
    models_to_test = {
        'Exponential Smoothing': ExponentialSmoothingModel,
        'Polynomial Trend': PolynomialTrendModel,
        'ARIMA Simple': ARIMASimpleModel
    }
    
    # GA parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'mutation_rate': 0.15,
        'crossover_rate': 0.85
    }
    
    # Optimize each model
    print(f"\n2. Optimizing models using genetic algorithm...")
    results = {}
    
    for model_name, model_class in models_to_test.items():
        try:
            result = optimize_model(
                model_class, train_data, test_data, 
                model_name, ga_params, metric='RMSE'
            )
            results[model_name] = result
            
            # Print summary
            summary = create_results_summary(
                model_name, result['parameters'], 
                result['metrics'], result['ga_generations'],
                result['execution_time']
            )
            print(summary)
            
        except Exception as e:
            print(f"Error optimizing {model_name}: {str(e)}")
            continue
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    
    # Plot original data and forecasts
    fig1 = plt.figure(figsize=(15, 10))
    
    # Original data plot
    plt.subplot(2, 2, 1)
    plt.plot(data, label='Original Data', color='blue', alpha=0.7)
    plt.axvline(x=len(train_data), color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
    plt.title(f'Original Time Series: {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Individual model predictions
    subplot_idx = 2
    for model_name, result in results.items():
        plt.subplot(2, 2, subplot_idx)
        
        # Plot training and test data
        plt.plot(range(len(train_data)), train_data, 
                label='Training Data', color='blue', alpha=0.7)
        plt.plot(range(len(train_data), len(data)), test_data, 
                label='Test Data', color='green', alpha=0.7)
        
        # Plot predictions
        pred_x = range(len(train_data), len(train_data) + len(result['predictions']))
        plt.plot(pred_x, result['predictions'], 
                label='Predictions', color='red', linewidth=2, linestyle='--')
        
        plt.title(f'{model_name} Forecast\nRMSE: {result["metrics"]["RMSE"]:.3f}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        subplot_idx += 1
        if subplot_idx > 4:  # Only show first 3 models
            break
    
    plt.tight_layout()
    plt.savefig('forecasting_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot GA evolution for best model
    if results:
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        fig2 = plot_ga_evolution(results[best_model_name]['fitness_history'])
        fig2.suptitle(f'GA Evolution - {best_model_name}')
        plt.savefig('ga_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Compare model performance
    if len(results) > 1:
        fig3 = compare_models_performance(results)
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Final summary
    print("\n4. Final Results Summary:")
    print("="*60)
    if results:
        best_model = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        print(f"Best performing model: {best_model}")
        print(f"Best RMSE: {results[best_model]['metrics']['RMSE']:.4f}")
        
        print("\nAll models performance (RMSE):")
        for model_name, result in sorted(results.items(), 
                                       key=lambda x: x[1]['metrics']['RMSE']):
            print(f"  {model_name:20s}: {result['metrics']['RMSE']:8.4f}")
    else:
        print("No models were successfully optimized.")
    
    print("\n5. Files generated:")
    print("  - forecasting_results.png: Main forecasting results")
    print("  - ga_evolution.png: Genetic algorithm evolution")
    print("  - model_comparison.png: Model performance comparison")
    
    print("\nDemonstration completed!")

if __name__ == "__main__":
    main()