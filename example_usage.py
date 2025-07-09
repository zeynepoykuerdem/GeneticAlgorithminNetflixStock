#!/usr/bin/env python3
"""
Example usage of GeneticNeuralOptimizer for stock price prediction

This script demonstrates how to use the genetic algorithm to optimize
neural network hyperparameters for time series prediction.
"""

from genetic_neural_optimizer import GeneticNeuralOptimizer
import pandas as pd
import numpy as np

def main():
    print("=== Genetic Algorithm Neural Network Optimizer ===\n")
    
    # Initialize the genetic optimizer with custom parameters
    ga_optimizer = GeneticNeuralOptimizer(
        population_size=8,      # Small population for faster execution
        crossover_rate=0.8,     # 80% crossover probability
        mutation_rate=0.15,     # 15% mutation probability
        generations=3,          # Few generations for demo
        sequence_length=30      # 30 days of history to predict next day
    )
    
    print("1. Loading and preparing data...")
    
    # Option 1: Use sample data (default)
    success = ga_optimizer.load_and_prepare_data()
    
    # Option 2: Use your own CSV file (uncomment and modify path)
    # success = ga_optimizer.load_and_prepare_data(
    #     file_path="your_stock_data.csv",
    #     features=['Close', 'Volume', 'High', 'Low'],  # Input features
    #     target=['Close']  # What to predict
    # )
    
    if not success:
        print("Failed to load data. Exiting...")
        return
    
    print("\n2. Running genetic algorithm evolution...")
    # This will take some time as each individual requires training a neural network
    best_individual = ga_optimizer.run_evolution()
    
    print("\n3. Plotting evolution history...")
    ga_optimizer.plot_evolution_history()
    
    print("\n4. Training the best model with more epochs...")
    best_model, history = ga_optimizer.train_best_model(epochs=50)
    
    print("\n5. Making predictions and plotting results...")
    predictions = ga_optimizer.make_predictions(best_model, plot_results=True)
    
    print("\n=== Results Summary ===")
    print(f"Best neural network configuration:")
    for key, value in best_individual.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest fitness score: {ga_optimizer.best_fitness:.6f}")
    print("Optimization complete! Check the generated plots for results.")


def run_with_real_data(csv_file_path):
    """
    Example of using the optimizer with real stock data
    
    Args:
        csv_file_path (str): Path to your CSV file with stock data
    """
    print(f"=== Using Real Data: {csv_file_path} ===\n")
    
    # Initialize with more aggressive parameters for real data
    ga_optimizer = GeneticNeuralOptimizer(
        population_size=15,
        crossover_rate=0.8,
        mutation_rate=0.1,
        generations=8,
        sequence_length=60  # 60 days of history
    )
    
    # Load your data
    success = ga_optimizer.load_and_prepare_data(
        file_path=csv_file_path,
        features=['Close', 'Volume', 'High', 'Low', 'Open'],  # Multiple features
        target=['Close']
    )
    
    if success:
        best_individual = ga_optimizer.run_evolution()
        ga_optimizer.plot_evolution_history()
        best_model, history = ga_optimizer.train_best_model(epochs=100)
        predictions = ga_optimizer.make_predictions(best_model)
        
        print(f"Best configuration: {best_individual}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run with your own data file
    # run_with_real_data("path/to/your/stock_data.csv")