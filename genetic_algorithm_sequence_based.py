from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    Sequence-based Genetic Algorithm for Time Series Prediction
    
    Chromosome structure: [weight1, weight2, ..., weightN] where N = sequence_length
    Each gene represents the importance/weight of that time step in the sequence
    """

    def __init__(self, generation_size, crossover_p, mutation_p, sequence_length, generations=20):
        self.generation_size = generation_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.sequence_length = sequence_length  # Chromosome length = sequence length
        self.generations = generations
        
        self.population = []
        self.fitness_scores = []
        self.best_chromosome = None
        self.best_fitness = 0
        self.best_score = float('inf')
        
        # Model parameters (can be fixed or also evolved)
        self.model_type = 'MLPRegressor'  # or 'RandomForest'
        self.hidden_layers = (100, 50)
        self.learning_rate = 0.001

    def load_data(self):
        """Load sample time series data"""
        print("Loading sample time series data...")
        # Create synthetic time series data
        np.random.seed(42)
        time_steps = 1000
        
        # Create trend + seasonality + noise
        t = np.arange(time_steps)
        trend = 0.02 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(2 * np.pi * t / 20)
        noise = np.random.normal(0, 2, time_steps)
        
        # Create multiple features
        feature1 = trend + seasonal + noise
        feature2 = 0.8 * feature1 + np.random.normal(0, 1, time_steps)
        feature3 = np.cumsum(np.random.normal(0, 0.5, time_steps))
        feature4 = feature1 * 0.3 + feature3 * 0.7
        target = feature1 * 0.6 + feature2 * 0.3 + feature3 * 0.1 + np.random.normal(0, 1, time_steps)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': feature1,
            'High': feature2, 
            'Low': feature3,
            'Volume': feature4,
            'Close': feature1 + np.random.normal(0, 0.5, time_steps),
            'Adj_Close': target
        })
        
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
        return X_scaled, selected_feature_names

    def create_sequences(self, features, target, sequence_length):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        X, y = np.array(X), np.array(y)
        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def apply_chromosome_weights(self, X_sequences, chromosome):
        """
        Apply chromosome weights to sequences
        Each gene in chromosome represents weight for that time step
        """
        # Normalize chromosome weights
        weights = np.array(chromosome)
        weights = weights / (np.sum(weights) + 1e-8)  # Normalize to sum to 1
        
        # Apply weights to each sequence
        # X_sequences shape: (n_samples, sequence_length, n_features)
        weighted_sequences = X_sequences.copy()
        
        for i in range(len(weights)):
            weighted_sequences[:, i, :] *= weights[i]
        
        return weighted_sequences

    def create_model(self):
        """Create model for prediction"""
        if self.model_type == 'MLPRegressor':
            return MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                learning_rate_init=self.learning_rate,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.model_type == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def train_model(self, X_sequences, y, chromosome):
        """Train model with chromosome-weighted sequences"""
        try:
            # Apply chromosome weights to sequences
            X_weighted = self.apply_chromosome_weights(X_sequences, chromosome)
            
            # Flatten sequences for traditional ML models
            # Shape: (n_samples, sequence_length * n_features)
            X_flattened = X_weighted.reshape(X_weighted.shape[0], -1)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_flattened, y, test_size=0.2, random_state=42
            )
            
            # Create and train model
            model = self.create_model()
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            return mse if not np.isnan(mse) else float('inf')
            
        except Exception as e:
            print(f"Error training model: {e}")
            return float('inf')

    def fitness_function(self, score):
        """Calculate fitness from score (lower score = higher fitness)"""
        if score == float('inf') or score <= 0:
            return 0
        return 1 / (1 + score)

    def create_initial_population(self):
        """Create initial random population where each chromosome has sequence_length genes"""
        population = []
        for _ in range(self.generation_size):
            # Create chromosome with length = sequence_length
            # Each gene represents weight/importance of that time step
            chromosome = [random.uniform(0.1, 1.0) for _ in range(self.sequence_length)]
            population.append(chromosome)
        
        print(f"Created initial population: {len(population)} chromosomes of length {self.sequence_length}")
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
        """Mutate chromosome weights"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_p:
                # Add small random change to weight
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = max(0.01, min(2.0, mutated[i]))  # Keep in reasonable range
        
        return mutated

    def run(self, X_sequences, y):
        """Run the genetic algorithm"""
        print(f"Starting Genetic Algorithm:")
        print(f"- Population size: {self.generation_size}")
        print(f"- Chromosome length: {self.sequence_length} (= sequence length)")
        print(f"- Generations: {self.generations}")
        print(f"- Crossover probability: {self.crossover_p}")
        print(f"- Mutation probability: {self.mutation_p}")
        
        # Create initial population
        self.population = self.create_initial_population()
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate fitness for each chromosome
            fitness_scores = []
            for i, chromosome in enumerate(self.population):
                score = self.train_model(X_sequences, y, chromosome)
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
            
            # Show best chromosome weights (first 10 time steps)
            if self.best_chromosome:
                weights_preview = self.best_chromosome[:min(10, len(self.best_chromosome))]
                print(f"Best weights preview: {[f'{w:.3f}' for w in weights_preview]}")
            
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
        print(f"Chromosome length: {len(self.best_chromosome)}")
        print(f"Score: {self.best_score:.4f}")
        print(f"Fitness: {self.best_fitness:.4f}")
        
        # Show chromosome weights analysis
        self.analyze_best_chromosome()
        
        return self.best_chromosome

    def analyze_best_chromosome(self):
        """Analyze the best chromosome weights"""
        if self.best_chromosome is None:
            return
        
        weights = np.array(self.best_chromosome)
        normalized_weights = weights / np.sum(weights)
        
        print(f"\nChromosome Analysis:")
        print(f"- Min weight: {np.min(weights):.4f}")
        print(f"- Max weight: {np.max(weights):.4f}")
        print(f"- Mean weight: {np.mean(weights):.4f}")
        print(f"- Std weight: {np.std(weights):.4f}")
        
        # Find most important time steps
        important_steps = np.argsort(normalized_weights)[-5:][::-1]
        print(f"Most important time steps:")
        for i, step in enumerate(important_steps):
            print(f"  {i+1}. Time step {step}: weight = {normalized_weights[step]:.4f}")

    def evaluate_best_model(self, X_sequences, y):
        """Train and evaluate the best model found"""
        if self.best_chromosome is None:
            print("No best chromosome found. Run the algorithm first.")
            return
        
        # Apply best chromosome weights
        X_weighted = self.apply_chromosome_weights(X_sequences, self.best_chromosome)
        X_flattened = X_weighted.reshape(X_weighted.shape[0], -1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_flattened, y, test_size=0.2, random_state=42
        )
        
        model = self.create_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal model evaluation:")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        return model, mse, r2

    def plot_chromosome_weights(self):
        """Plot the chromosome weights to visualize time step importance"""
        if self.best_chromosome is None:
            print("No best chromosome found. Run the algorithm first.")
            return
        
        plt.figure(figsize=(12, 6))
        weights = np.array(self.best_chromosome)
        normalized_weights = weights / np.sum(weights)
        
        plt.subplot(1, 2, 1)
        plt.plot(weights, 'b-', marker='o', markersize=3)
        plt.title('Raw Chromosome Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(normalized_weights, 'r-', marker='o', markersize=3)
        plt.title('Normalized Chromosome Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Weight')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chromosome_weights.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Chromosome weights plot saved as 'chromosome_weights.png'")


def main():
    """Main function to run the sequence-based genetic algorithm"""
    print("=== Sequence-based Genetic Algorithm for Time Series Prediction ===")
    
    # Define sequence length
    sequence_length = 60  # Your sequence length
    
    # Initialize GA with chromosome length = sequence length
    ga = GeneticAlgorithm(
        generation_size=20,
        crossover_p=0.8,
        mutation_p=0.2,
        sequence_length=sequence_length,  # Chromosome length = sequence length
        generations=15
    )
    
    # Load and preprocess data
    data = ga.load_data()
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    target = ['Adj_Close']

    X_scaled, y_scaled, scaler_X, scaler_y = ga.pre_process_data(data, features, target)
    print(f"Data preprocessed. X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")

    # Feature selection
    X_selected, selected_features = ga.feature_selection(X_scaled, y_scaled, features, k=3)
    print(f"Selected features shape: {X_selected.shape}")

    # Create sequences - Bu adımdan sonra chromosome length = sequence length
    X_sequences, y_sequences = ga.create_sequences(X_selected, y_scaled, sequence_length)
    print(f"Sequence length: {sequence_length}")
    print(f"Chromosome length will be: {sequence_length}")

    # Run genetic algorithm
    best_chromosome = ga.run(X_sequences, y_sequences)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    model, mse, r2 = ga.evaluate_best_model(X_sequences, y_sequences)
    
    # Plot chromosome weights
    ga.plot_chromosome_weights()


if __name__ == "__main__":
    main()