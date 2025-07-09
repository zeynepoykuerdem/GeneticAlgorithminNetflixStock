import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import random
import warnings
warnings.filterwarnings('ignore')

class GeneticSklearnOptimizer:
    def __init__(self, population_size=20, crossover_rate=0.8, mutation_rate=0.1, 
                 generations=10, sequence_length=30):
        """
        Initialize the Genetic Algorithm for Scikit-learn Model Optimization
        
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
            
            # Add technical indicators
            data = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates)),
                'High': np.array(prices) * (1 + np.random.uniform(0, 0.05, len(dates))),
                'Low': np.array(prices) * (1 - np.random.uniform(0, 0.05, len(dates))),
                'Open': np.array(prices) * (1 + np.random.uniform(-0.02, 0.02, len(dates)))
            })
            
            # Add moving averages
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data.dropna(inplace=True)
            
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
        available_features = ['Close', 'Volume', 'High', 'Low', 'Open', 'MA_5', 'MA_20', 'RSI']
        features = [f for f in available_features if f in data.columns]
        
        if not features:
            features = ['Close']
        
        print(f"Using features: {features}")
        
        # Preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        X_scaled = self.scaler_X.fit_transform(data[features])
        y_scaled = self.scaler_y.fit_transform(data[target])
        
        # Create sequences for time series
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled.flatten())
        
        # Train/validation split (80/20)
        split_idx = int(len(X_sequences) * 0.8)
        self.X_train, self.X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        self.y_train, self.y_val = y_sequences[:split_idx], y_sequences[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}, {self.y_val.shape}")
        
        return True
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            # Flatten the sequence to create feature vector
            X_seq.append(X[i-self.sequence_length:i].flatten())
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _create_chromosome(self):
        """
        Create a random chromosome representing model hyperparameters
        
        Returns:
            dict: Chromosome with model parameters
        """
        model_type = random.choice(['RandomForest', 'GradientBoosting', 'SVR', 'Ridge', 'MLP'])
        
        chromosome = {'model_type': model_type}
        
        if model_type == 'RandomForest':
            chromosome.update({
                'n_estimators': random.randint(50, 200),
                'max_depth': random.choice([None, 5, 10, 15, 20]),
                'min_samples_split': random.randint(2, 10),
                'min_samples_leaf': random.randint(1, 5)
            })
        elif model_type == 'GradientBoosting':
            chromosome.update({
                'n_estimators': random.randint(50, 200),
                'learning_rate': random.uniform(0.01, 0.3),
                'max_depth': random.randint(3, 10),
                'subsample': random.uniform(0.7, 1.0)
            })
        elif model_type == 'SVR':
            chromosome.update({
                'C': random.uniform(0.1, 100),
                'epsilon': random.uniform(0.01, 1.0),
                'kernel': random.choice(['linear', 'rbf', 'poly']),
                'gamma': random.choice(['scale', 'auto'])
            })
        elif model_type == 'Ridge':
            chromosome.update({
                'alpha': random.uniform(0.1, 100),
                'solver': random.choice(['auto', 'svd', 'cholesky', 'lsqr'])
            })
        elif model_type == 'MLP':
            hidden_layers = random.randint(1, 3)
            chromosome.update({
                'hidden_layer_sizes': tuple(random.randint(50, 200) for _ in range(hidden_layers)),
                'alpha': random.uniform(0.0001, 0.01),
                'learning_rate': random.choice(['constant', 'invscaling', 'adaptive']),
                'max_iter': random.randint(200, 1000)
            })
        
        return chromosome
    
    def _create_model(self, chromosome):
        """
        Create a model based on chromosome parameters
        
        Args:
            chromosome (dict): Chromosome containing model parameters
            
        Returns:
            sklearn model: Configured model
        """
        model_type = chromosome['model_type']
        params = {k: v for k, v in chromosome.items() if k != 'model_type'}
        
        if model_type == 'RandomForest':
            return RandomForestRegressor(**params, random_state=42)
        elif model_type == 'GradientBoosting':
            return GradientBoostingRegressor(**params, random_state=42)
        elif model_type == 'SVR':
            return SVR(**params)
        elif model_type == 'Ridge':
            return Ridge(**params)
        elif model_type == 'MLP':
            return MLPRegressor(**params, random_state=42, max_iter=min(params.get('max_iter', 500), 500))
        
        return RandomForestRegressor(random_state=42)  # fallback
    
    def _evaluate_fitness(self, chromosome):
        """
        Evaluate the fitness of a chromosome by training and validating the model
        
        Args:
            chromosome (dict): Chromosome to evaluate
            
        Returns:
            float: Fitness score (negative mean squared error for maximization)
        """
        try:
            # Create and train model
            model = self._create_model(chromosome)
            
            # Use cross-validation for more robust evaluation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=3, scoring='neg_mean_squared_error')
            
            # Return average cross-validation score
            fitness = np.mean(cv_scores)
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating chromosome {chromosome['model_type']}: {e}")
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
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
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
        
        # If different model types, return parents unchanged
        if parent1['model_type'] != parent2['model_type']:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Crossover parameters (excluding model_type)
        for key in parent1.keys():
            if key != 'model_type' and random.random() < 0.5:
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
            # Sometimes change the entire model type
            if random.random() < 0.3:
                return self._create_chromosome()
            
            # Otherwise mutate existing parameters
            model_type = chromosome['model_type']
            
            if model_type == 'RandomForest':
                param = random.choice(['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])
                if param == 'n_estimators':
                    mutated[param] = random.randint(50, 200)
                elif param == 'max_depth':
                    mutated[param] = random.choice([None, 5, 10, 15, 20])
                elif param == 'min_samples_split':
                    mutated[param] = random.randint(2, 10)
                elif param == 'min_samples_leaf':
                    mutated[param] = random.randint(1, 5)
                    
            elif model_type == 'GradientBoosting':
                param = random.choice(['n_estimators', 'learning_rate', 'max_depth', 'subsample'])
                if param == 'n_estimators':
                    mutated[param] = random.randint(50, 200)
                elif param == 'learning_rate':
                    mutated[param] = random.uniform(0.01, 0.3)
                elif param == 'max_depth':
                    mutated[param] = random.randint(3, 10)
                elif param == 'subsample':
                    mutated[param] = random.uniform(0.7, 1.0)
                    
            # Add mutations for other model types...
        
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
                print(f"Individual {i+1} ({chromosome['model_type']}): Fitness = {fitness:.6f}")
            
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
        plt.ylabel('Fitness (Negative MSE)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.best_fitness_history, 'g-', linewidth=2)
        plt.title('Best Fitness Progress')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sklearn_evolution_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_best_model(self):
        """
        Train the best model found by the genetic algorithm
        
        Returns:
            sklearn model: Trained model
        """
        if self.best_individual is None:
            print("Error: No best individual found. Please run evolution first.")
            return None
        
        print("Training best model...")
        print(f"Best parameters: {self.best_individual}")
        
        # Create and train the best model
        model = self._create_model(self.best_individual)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mse = mean_squared_error(self.y_train, train_pred)
        val_mse = mean_squared_error(self.y_val, val_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        
        print(f"Training MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"Validation MSE: {val_mse:.6f}, R²: {val_r2:.4f}")
        
        return model
    
    def make_predictions(self, model, plot_results=True):
        """
        Make predictions using the trained model
        
        Args:
            model: Trained sklearn model
            plot_results (bool): Whether to plot the results
            
        Returns:
            tuple: Actual and predicted values
        """
        # Make predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, train_pred)
        val_mse = mean_squared_error(self.y_val, val_pred)
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        
        print(f"Training - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"Validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        if plot_results:
            plt.figure(figsize=(15, 8))
            
            # Plot training predictions
            plt.subplot(2, 1, 1)
            plt.scatter(self.y_train[:200], train_pred[:200], alpha=0.6, label='Predictions')
            plt.plot([self.y_train[:200].min(), self.y_train[:200].max()], 
                    [self.y_train[:200].min(), self.y_train[:200].max()], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Training Predictions (First 200 samples)')
            plt.legend()
            plt.grid(True)
            
            # Plot validation predictions
            plt.subplot(2, 1, 2)
            plt.scatter(self.y_val, val_pred, alpha=0.6, label='Predictions')
            plt.plot([self.y_val.min(), self.y_val.max()], 
                    [self.y_val.min(), self.y_val.max()], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Validation Predictions')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('sklearn_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return (self.y_train, train_pred, self.y_val, val_pred)


# Example usage
if __name__ == "__main__":
    print("=== Genetic Algorithm Scikit-learn Optimizer ===\n")
    
    # Initialize the genetic optimizer
    ga_optimizer = GeneticSklearnOptimizer(
        population_size=8,  # Small for demonstration
        crossover_rate=0.8,
        mutation_rate=0.15,
        generations=5,  # Small for demonstration
        sequence_length=20
    )
    
    # Load and prepare data
    success = ga_optimizer.load_and_prepare_data()
    
    if success:
        # Run evolution
        best_individual = ga_optimizer.run_evolution()
        
        # Plot evolution history
        ga_optimizer.plot_evolution_history()
        
        # Train the best model
        best_model = ga_optimizer.train_best_model()
        
        # Make predictions and plot results
        predictions = ga_optimizer.make_predictions(best_model)
        
        print("\nGenetic Algorithm Optimization Complete!")
        print(f"Best configuration found: {best_individual}")
    else:
        print("Failed to load data. Exiting...")