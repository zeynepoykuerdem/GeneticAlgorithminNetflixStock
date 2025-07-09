import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import warnings
warnings.filterwarnings('ignore')

class GeneticNeuralOptimizer:
    def __init__(self, population_size=20, crossover_rate=0.8, mutation_rate=0.1, 
                 generations=10, sequence_length=60):
        """
        Initialize the Genetic Algorithm for Neural Network Optimization
        
        Args:
            population_size (int): Number of individuals in each generation
            crossover_rate (float): Probability of crossover
            mutation_rate (float): Probability of mutation
            generations (int): Number of generations to evolve
            sequence_length (int): Length of input sequences for time series
        """
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.sequence_length = sequence_length
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.scaler_X = None
        self.scaler_y = None
        
        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def load_and_prepare_data(self, file_path=None, features=['Close'], target=['Close']):
        """
        Load and prepare time series data for training
        
        Args:
            file_path (str): Path to CSV file. If None, generates sample data
            features (list): Feature columns to use
            target (list): Target columns to predict
        """
        if file_path is None:
            # Generate sample stock data for demonstration
            print("No file path provided. Generating sample stock data...")
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic stock price data with trend and noise
            base_price = 100
            trend = 0.0002
            prices = []
            price = base_price
            
            for i in range(len(dates)):
                # Add trend, seasonality, and random walk
                daily_return = np.random.normal(trend, 0.02)
                price *= (1 + daily_return)
                prices.append(price)
            
            data = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates)),
                'High': np.array(prices) * (1 + np.random.uniform(0, 0.05, len(dates))),
                'Low': np.array(prices) * (1 - np.random.uniform(0, 0.05, len(dates))),
                'Open': np.array(prices) * (1 + np.random.uniform(-0.02, 0.02, len(dates)))
            })
        else:
            try:
                data = pd.read_csv(file_path)
                print(f"Loaded data from {file_path}")
            except Exception as e:
                print(f"Error loading data: {e}")
                return False
        
        print(f"Data shape: {data.shape}")
        print(f"Available columns: {data.columns.tolist()}")
        
        # Ensure features and target exist
        missing_features = [f for f in features if f not in data.columns]
        missing_target = [t for t in target if t not in data.columns]
        
        if missing_features:
            print(f"Warning: Features {missing_features} not found. Using 'Close' only.")
            features = ['Close']
        if missing_target:
            print(f"Warning: Target {missing_target} not found. Using 'Close' only.")
            target = ['Close']
        
        # Preprocessing
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        X_scaled = self.scaler_X.fit_transform(data[features])
        y_scaled = self.scaler_y.fit_transform(data[target])
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled)
        
        # Train/validation split (80/20)
        split_idx = int(len(X_sequences) * 0.8)
        self.X_train, self.X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        self.y_train, self.y_val = y_sequences[:split_idx], y_sequences[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}, {self.y_val.shape}")
        
        return True
    
    def _create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _create_chromosome(self):
        """
        Create a random chromosome representing neural network hyperparameters
        
        Returns:
            dict: Chromosome with neural network parameters
        """
        chromosome = {
            'layer_type': random.choice(['LSTM', 'GRU', 'Bidirectional_LSTM', 'Bidirectional_GRU']),
            'num_layers': random.randint(1, 3),
            'units': [random.randint(32, 128) for _ in range(3)],  # Max 3 layers
            'dropout_rate': random.uniform(0.1, 0.5),
            'learning_rate': random.uniform(0.0001, 0.01),
            'batch_size': random.choice([16, 32, 64]),
            'optimizer': random.choice(['adam', 'rmsprop'])
        }
        return chromosome
    
    def _create_model(self, chromosome):
        """
        Create a neural network model based on chromosome parameters
        
        Args:
            chromosome (dict): Chromosome containing model parameters
            
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = Sequential()
        
        # Input layer
        input_shape = (self.sequence_length, self.X_train.shape[2])
        
        # Add recurrent layers
        for i in range(chromosome['num_layers']):
            units = chromosome['units'][i]
            return_sequences = (i < chromosome['num_layers'] - 1)
            
            if chromosome['layer_type'] == 'LSTM':
                model.add(LSTM(units, return_sequences=return_sequences, 
                             input_shape=input_shape if i == 0 else None))
            elif chromosome['layer_type'] == 'GRU':
                model.add(GRU(units, return_sequences=return_sequences,
                            input_shape=input_shape if i == 0 else None))
            elif chromosome['layer_type'] == 'Bidirectional_LSTM':
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences),
                                      input_shape=input_shape if i == 0 else None))
            elif chromosome['layer_type'] == 'Bidirectional_GRU':
                model.add(Bidirectional(GRU(units, return_sequences=return_sequences),
                                      input_shape=input_shape if i == 0 else None))
            
            # Add dropout
            model.add(Dropout(chromosome['dropout_rate']))
        
        # Output layer
        model.add(Dense(self.y_train.shape[1]))
        
        # Compile model
        optimizer = Adam(learning_rate=chromosome['learning_rate']) if chromosome['optimizer'] == 'adam' else 'rmsprop'
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _evaluate_fitness(self, chromosome):
        """
        Evaluate the fitness of a chromosome by training and validating the model
        
        Args:
            chromosome (dict): Chromosome to evaluate
            
        Returns:
            float: Fitness score (negative validation loss for maximization)
        """
        try:
            # Create and train model
            model = self._create_model(chromosome)
            
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Train model
            history = model.fit(
                self.X_train, self.y_train,
                batch_size=chromosome['batch_size'],
                epochs=20,  # Limited epochs for faster evaluation
                validation_data=(self.X_val, self.y_val),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Get validation loss (lower is better, so we negate for maximization)
            val_loss = min(history.history['val_loss'])
            fitness = -val_loss
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating chromosome: {e}")
            return float('-inf')
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """
        Tournament selection for choosing parents
        
        Args:
            population (list): List of chromosomes
            fitness_scores (list): Corresponding fitness scores
            tournament_size (int): Size of tournament
            
        Returns:
            dict: Selected chromosome
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parent chromosomes
        
        Args:
            parent1 (dict): First parent chromosome
            parent2 (dict): Second parent chromosome
            
        Returns:
            tuple: Two offspring chromosomes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Crossover for different parameter types
        for key in parent1.keys():
            if random.random() < 0.5:
                if key == 'units':
                    # For list parameters, mix elements
                    for i in range(len(child1[key])):
                        if random.random() < 0.5:
                            child1[key][i], child2[key][i] = child2[key][i], child1[key][i]
                else:
                    # For scalar parameters, swap
                    child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(self, chromosome):
        """
        Perform mutation on a chromosome
        
        Args:
            chromosome (dict): Chromosome to mutate
            
        Returns:
            dict: Mutated chromosome
        """
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(list(chromosome.keys()))
            
            if mutation_type == 'layer_type':
                mutated['layer_type'] = random.choice(['LSTM', 'GRU', 'Bidirectional_LSTM', 'Bidirectional_GRU'])
            elif mutation_type == 'num_layers':
                mutated['num_layers'] = random.randint(1, 3)
            elif mutation_type == 'units':
                layer_to_mutate = random.randint(0, len(mutated['units']) - 1)
                mutated['units'][layer_to_mutate] = random.randint(32, 128)
            elif mutation_type == 'dropout_rate':
                mutated['dropout_rate'] = random.uniform(0.1, 0.5)
            elif mutation_type == 'learning_rate':
                mutated['learning_rate'] = random.uniform(0.0001, 0.01)
            elif mutation_type == 'batch_size':
                mutated['batch_size'] = random.choice([16, 32, 64])
            elif mutation_type == 'optimizer':
                mutated['optimizer'] = random.choice(['adam', 'rmsprop'])
        
        return mutated
    
    def run_evolution(self):
        """
        Run the genetic algorithm evolution process
        
        Returns:
            dict: Best chromosome found
        """
        if self.X_train is None:
            print("Error: No data loaded. Please call load_and_prepare_data() first.")
            return None
        
        print(f"Starting evolution with {self.population_size} individuals for {self.generations} generations")
        
        # Initialize population
        population = [self._create_chromosome() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate fitness for all individuals
            fitness_scores = []
            for i, chromosome in enumerate(population):
                fitness = self._evaluate_fitness(chromosome)
                fitness_scores.append(fitness)
                print(f"Individual {i+1}: Fitness = {fitness:.6f}")
            
            # Track best and average fitness
            best_fitness_gen = max(fitness_scores)
            avg_fitness_gen = np.mean(fitness_scores)
            
            self.best_fitness_history.append(best_fitness_gen)
            self.avg_fitness_history.append(avg_fitness_gen)
            
            # Update best individual
            if best_fitness_gen > self.best_fitness:
                self.best_fitness = best_fitness_gen
                best_idx = fitness_scores.index(best_fitness_gen)
                self.best_individual = population[best_idx].copy()
            
            print(f"Best fitness: {best_fitness_gen:.6f}, Average fitness: {avg_fitness_gen:.6f}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(population[best_idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
        
        print(f"\nEvolution completed!")
        print(f"Best fitness achieved: {self.best_fitness:.6f}")
        print(f"Best individual: {self.best_individual}")
        
        return self.best_individual
    
    def plot_evolution_history(self):
        """Plot the evolution history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(self.avg_fitness_history, 'r--', label='Average Fitness', linewidth=2)
        plt.title('Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.best_fitness_history, 'g-', linewidth=2)
        plt.title('Best Fitness Progress')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evolution_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_best_model(self, epochs=100):
        """
        Train the best model found by the genetic algorithm
        
        Args:
            epochs (int): Number of training epochs
            
        Returns:
            tf.keras.Model: Trained model
        """
        if self.best_individual is None:
            print("Error: No best individual found. Please run evolution first.")
            return None
        
        print("Training best model...")
        print(f"Best parameters: {self.best_individual}")
        
        # Create and train the best model
        model = self._create_model(self.best_individual)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=self.best_individual['batch_size'],
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate model
        train_loss = model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss = model.evaluate(self.X_val, self.y_val, verbose=0)
        
        print(f"Final Training Loss: {train_loss[0]:.6f}")
        print(f"Final Validation Loss: {val_loss[0]:.6f}")
        
        return model, history
    
    def make_predictions(self, model, plot_results=True):
        """
        Make predictions using the trained model
        
        Args:
            model: Trained Keras model
            plot_results (bool): Whether to plot the results
            
        Returns:
            tuple: Actual and predicted values
        """
        # Make predictions
        train_pred = model.predict(self.X_train, verbose=0)
        val_pred = model.predict(self.X_val, verbose=0)
        
        # Inverse transform to original scale
        train_pred_rescaled = self.scaler_y.inverse_transform(train_pred)
        val_pred_rescaled = self.scaler_y.inverse_transform(val_pred)
        y_train_rescaled = self.scaler_y.inverse_transform(self.y_train)
        y_val_rescaled = self.scaler_y.inverse_transform(self.y_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_rescaled, train_pred_rescaled)
        val_mse = mean_squared_error(y_val_rescaled, val_pred_rescaled)
        train_mae = mean_absolute_error(y_train_rescaled, train_pred_rescaled)
        val_mae = mean_absolute_error(y_val_rescaled, val_pred_rescaled)
        
        print(f"Training MSE: {train_mse:.2f}, MAE: {train_mae:.2f}")
        print(f"Validation MSE: {val_mse:.2f}, MAE: {val_mae:.2f}")
        
        if plot_results:
            plt.figure(figsize=(15, 8))
            
            # Plot training predictions
            plt.subplot(2, 1, 1)
            plt.plot(y_train_rescaled[:200], label='Actual', alpha=0.7)
            plt.plot(train_pred_rescaled[:200], label='Predicted', alpha=0.7)
            plt.title('Training Predictions (First 200 samples)')
            plt.legend()
            plt.grid(True)
            
            # Plot validation predictions
            plt.subplot(2, 1, 2)
            plt.plot(y_val_rescaled, label='Actual', alpha=0.7)
            plt.plot(val_pred_rescaled, label='Predicted', alpha=0.7)
            plt.title('Validation Predictions')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return (y_train_rescaled, train_pred_rescaled, y_val_rescaled, val_pred_rescaled)


# Example usage
if __name__ == "__main__":
    # Initialize the genetic optimizer
    ga_optimizer = GeneticNeuralOptimizer(
        population_size=10,  # Small for demonstration
        crossover_rate=0.8,
        mutation_rate=0.15,
        generations=5,  # Small for demonstration
        sequence_length=30
    )
    
    # Load and prepare data (using sample data)
    success = ga_optimizer.load_and_prepare_data()
    
    if success:
        # Run evolution
        best_individual = ga_optimizer.run_evolution()
        
        # Plot evolution history
        ga_optimizer.plot_evolution_history()
        
        # Train the best model
        best_model, history = ga_optimizer.train_best_model(epochs=50)
        
        # Make predictions and plot results
        predictions = ga_optimizer.make_predictions(best_model)
        
        print("\nGenetic Algorithm Optimization Complete!")
        print(f"Best configuration found: {best_individual}")