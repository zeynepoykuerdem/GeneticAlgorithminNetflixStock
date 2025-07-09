from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing Machine Learning model hyperparameters
    
    Chromosome structure: [model_type, param1, param2, param3, ...]
    """

    def __init__(self, generation_size, crossover_p, mutation_p, generations=20):
        self.generation_size = generation_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.generations = generations
        
        # Define model types and their parameter ranges
        self.model_types = {
            0: 'RandomForest',
            1: 'GradientBoosting', 
            2: 'Ridge',
            3: 'Lasso',
            4: 'SVR',
            5: 'MLPRegressor'
        }
        
        self.population = []
        self.fitness_scores = []
        self.best_chromosome = None
        self.best_fitness = 0
        self.best_score = float('inf')

    def load_data(self):
        """Load sample regression data (since Kaggle data isn't available)"""
        print("Loading sample stock-like data...")
        # Create synthetic stock-like data
        X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
        
        # Convert to DataFrame with stock-like column names
        feature_names = ['Open', 'High', 'Low', 'Volume', 'Close']
        data = pd.DataFrame(X, columns=feature_names)
        data['Adj_Close'] = y
        
        print(f"Data shape: {data.shape}")
        return data

    def pre_process_data(self, data, features, target):
        """Normalize data to have same range"""
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_scaled = scaler_X.fit_transform(data[features])
        y_scaled = scaler_y.fit_transform(data[target].values.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled, scaler_X, scaler_y

    def feature_selection(self, X_scaled, y_scaled, features, k=3):
        """Select best k features"""
        fs = SelectKBest(score_func=f_regression, k=k)
        fs.fit(X_scaled, y_scaled)

        selected_feature_mask = fs.get_support()  
        selected_feature_names = np.array(features)[selected_feature_mask]

        print("Selected features:")
        print(selected_feature_names)

        X_scaled = X_scaled[:, selected_feature_mask]
        return X_scaled

    def decode_chromosome(self, chromosome):
        """Decode chromosome to model parameters"""
        model_type = chromosome[0]
        params = chromosome[1:]
        
        if model_type == 0:  # RandomForest
            return {
                'model': 'RandomForest',
                'n_estimators': int(params[0] * 200) + 10,  # 10-210
                'max_depth': int(params[1] * 20) + 1,       # 1-21
                'min_samples_split': int(params[2] * 10) + 2, # 2-12
                'min_samples_leaf': int(params[3] * 5) + 1    # 1-6
            }
        elif model_type == 1:  # GradientBoosting
            return {
                'model': 'GradientBoosting',
                'n_estimators': int(params[0] * 200) + 10,    # 10-210
                'learning_rate': params[1] * 0.3 + 0.01,     # 0.01-0.31
                'max_depth': int(params[2] * 10) + 1,        # 1-11
                'subsample': params[3] * 0.5 + 0.5           # 0.5-1.0
            }
        elif model_type == 2:  # Ridge
            return {
                'model': 'Ridge',
                'alpha': params[0] * 10 + 0.1,               # 0.1-10.1
                'solver': ['auto', 'svd', 'cholesky', 'lsqr'][int(params[1] * 4) % 4]
            }
        elif model_type == 3:  # Lasso
            return {
                'model': 'Lasso',
                'alpha': params[0] * 10 + 0.1,               # 0.1-10.1
                'max_iter': int(params[1] * 1000) + 100      # 100-1100
            }
        elif model_type == 4:  # SVR
            return {
                'model': 'SVR',
                'C': params[0] * 100 + 0.1,                  # 0.1-100.1
                'epsilon': params[1] * 1.0 + 0.01,          # 0.01-1.01
                'gamma': ['scale', 'auto'][int(params[2] * 2) % 2]
            }
        elif model_type == 5:  # MLPRegressor
            return {
                'model': 'MLPRegressor',
                'hidden_layer_sizes': (int(params[0] * 200) + 10,),  # (10-210,)
                'alpha': params[1] * 0.01 + 0.0001,                  # 0.0001-0.0101
                'learning_rate_init': params[2] * 0.01 + 0.001,      # 0.001-0.011
                'max_iter': int(params[3] * 500) + 200                # 200-700
            }

    def create_model(self, params):
        """Create model based on decoded parameters"""
        try:
            if params['model'] == 'RandomForest':
                return RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    random_state=42
                )
            elif params['model'] == 'GradientBoosting':
                return GradientBoostingRegressor(
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    subsample=params['subsample'],
                    random_state=42
                )
            elif params['model'] == 'Ridge':
                return Ridge(
                    alpha=params['alpha'],
                    solver=params['solver']
                )
            elif params['model'] == 'Lasso':
                return Lasso(
                    alpha=params['alpha'],
                    max_iter=params['max_iter']
                )
            elif params['model'] == 'SVR':
                return SVR(
                    C=params['C'],
                    epsilon=params['epsilon'],
                    gamma=params['gamma']
                )
            elif params['model'] == 'MLPRegressor':
                return MLPRegressor(
                    hidden_layer_sizes=params['hidden_layer_sizes'],
                    alpha=params['alpha'],
                    learning_rate_init=params['learning_rate_init'],
                    max_iter=params['max_iter'],
                    random_state=42
                )
        except Exception as e:
            print(f"Error creating model: {e}")
            return RandomForestRegressor(random_state=42)

    def train_model(self, X_scaled, y_scaled, chromosome):
        """Train model and return score"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            params = self.decode_chromosome(chromosome)
            model = self.create_model(params)
            
            # Use cross-validation for more robust evaluation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                                      scoring='neg_mean_squared_error')
            score = -cv_scores.mean()  # Convert back to positive MSE
            
            return score if not np.isnan(score) else float('inf')
            
        except Exception as e:
            print(f"Error training model: {e}")
            return float('inf')

    def fitness_function(self, score):
        """Calculate fitness from score (lower score = higher fitness)"""
        if score == float('inf') or score <= 0:
            return 0
        return 1 / (1 + score)

    def create_initial_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.generation_size):
            # Create chromosome: [model_type, param1, param2, param3, param4]
            chromosome = [
                random.randint(0, len(self.model_types) - 1),  # model type
                random.random(),  # param1
                random.random(),  # param2
                random.random(),  # param3
                random.random()   # param4
            ]
            population.append(chromosome)
        
        return population

    def selection_process(self, population, fitness_scores, selection_type="tournament"):
        """Select parents for reproduction"""
        if selection_type == "tournament":
            tournament_size = 3
            parent1 = self.tournament_selection(population, fitness_scores, tournament_size)
            parent2 = self.tournament_selection(population, fitness_scores, tournament_size)
        else:  # roulette wheel
            parent1 = self.roulette_selection(population, fitness_scores)
            parent2 = self.roulette_selection(population, fitness_scores)
        
        return parent1, parent2

    def tournament_selection(self, population, fitness_scores, tournament_size):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda x: fitness_scores[x])
        return population[best_idx].copy()

    def roulette_selection(self, population, fitness_scores):
        """Roulette wheel selection"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population).copy()
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current > pick:
                return population[i].copy()
        return population[-1].copy()

    def crossover_process(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_p:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2

    def mutation_process(self, chromosome):
        """Mutate chromosome"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_p:
                if i == 0:  # model type
                    mutated[i] = random.randint(0, len(self.model_types) - 1)
                else:  # parameters
                    mutated[i] = random.random()
        
        return mutated

    def run(self, X_scaled, y_scaled):
        """Run the genetic algorithm"""
        print(f"Starting Genetic Algorithm with {self.generation_size} individuals for {self.generations} generations")
        
        # Create initial population
        self.population = self.create_initial_population()
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate fitness for each chromosome
            fitness_scores = []
            for i, chromosome in enumerate(self.population):
                score = self.train_model(X_scaled, y_scaled, chromosome)
                fitness = self.fitness_function(score)
                fitness_scores.append(fitness)
                
                # Track best chromosome
                if score < self.best_score:
                    self.best_score = score
                    self.best_fitness = fitness
                    self.best_chromosome = chromosome.copy()
            
            self.fitness_scores = fitness_scores
            avg_fitness = np.mean(fitness_scores)
            max_fitness = max(fitness_scores)
            
            print(f"Average fitness: {avg_fitness:.4f}, Best fitness: {max_fitness:.4f}")
            print(f"Best score so far: {self.best_score:.4f}")
            
            # Create new generation
            new_population = []
            
            # Elitism - keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(self.population[best_idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.generation_size:
                parent1, parent2 = self.selection_process(self.population, fitness_scores)
                child1, child2 = self.crossover_process(parent1, parent2)
                
                child1 = self.mutation_process(child1)
                child2 = self.mutation_process(child2)
                
                new_population.extend([child1, child2])
            
            # Keep population size constant
            self.population = new_population[:self.generation_size]
        
        print(f"\nBest solution found:")
        print(f"Chromosome: {self.best_chromosome}")
        print(f"Score: {self.best_score:.4f}")
        print(f"Fitness: {self.best_fitness:.4f}")
        
        # Decode and print best solution
        best_params = self.decode_chromosome(self.best_chromosome)
        print(f"Best model configuration:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_chromosome

    def evaluate_best_model(self, X_scaled, y_scaled):
        """Train and evaluate the best model found"""
        if self.best_chromosome is None:
            print("No best chromosome found. Run the algorithm first.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        params = self.decode_chromosome(self.best_chromosome)
        model = self.create_model(params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal model evaluation:")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        return model, mse, r2


def main():
    """Main function to run the genetic algorithm"""
    print("=== Genetic Algorithm for Machine Learning Model Optimization ===")
    
    # Initialize GA
    ga = GeneticAlgorithm(
        generation_size=20,
        crossover_p=0.8,
        mutation_p=0.2,
        generations=10
    )
    
    # Load and preprocess data
    data = ga.load_data()
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    target = ['Adj_Close']

    X_scaled, y_scaled, scaler_X, scaler_y = ga.pre_process_data(data, features, target)
    print(f"Data preprocessed. X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")

    # Feature selection
    X_selected = ga.feature_selection(X_scaled, y_scaled, features, k=3)
    print(f"Selected features shape: {X_selected.shape}")

    # Run genetic algorithm
    best_chromosome = ga.run(X_selected, y_scaled)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    model, mse, r2 = ga.evaluate_best_model(X_selected, y_scaled)


if __name__ == "__main__":
    main()