from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing Neural Network hyperparameters
    
    Chromosome structure: [num_layers, neurons_layer1, neurons_layer2, ..., learning_rate_idx, activation_idx]
    """

    def __init__(self, generation_size, crossover_p, mutation_p, max_layers=5, max_neurons=128, generations=20):
        self.generation_size = generation_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.generations = generations
        self.learning_rates = [0.001, 0.01, 0.1, 0.0001]
        self.activations = ['relu', 'tanh', 'sigmoid']
        self.population = []
        self.fitness_scores = []
        self.best_chromosome = None
        self.best_fitness = 0

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

    def create_sequences(self, features, target, sequence_length):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(features) - sequence_length - 1):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        X, y = np.array(X), np.array(y)
        return X, y

    def decode_chromosome(self, chromosome):
        """Decode chromosome to neural network parameters"""
        num_layers = chromosome[0]
        neurons = chromosome[1:num_layers+1]
        lr_idx = chromosome[-2]
        activation_idx = chromosome[-1]
        
        learning_rate = self.learning_rates[lr_idx]
        activation = self.activations[activation_idx]
        
        return num_layers, neurons, learning_rate, activation

    def model_builder(self, input_shape, chromosome):
        """Build neural network model based on chromosome"""
        num_layers, neurons, learning_rate, activation = self.decode_chromosome(chromosome)
        
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        for i in range(num_layers):
            model.add(Dense(neurons[i], activation=activation))
        
        model.add(Dense(1))
        
        # Compile with decoded learning rate
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def train_model(self, X_scaled, y_scaled, chromosome):
        """Train model and return loss"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            model = self.model_builder((X_train.shape[1],), chromosome)
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
            
            loss = model.evaluate(X_test, y_test, verbose=0)
            return loss
        except Exception as e:
            print(f"Error training model: {e}")
            return float('inf')

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss

    def fitness_function(self, loss_value):
        """Calculate fitness from loss (lower loss = higher fitness)"""
        if loss_value == float('inf') or loss_value <= 0:
            return 0
        return 1 / (1 + loss_value)

    def create_initial_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.generation_size):
            # Create chromosome: [num_layers, neurons_per_layer..., lr_idx, activation_idx]
            num_layers = random.randint(1, self.max_layers)
            chromosome = [num_layers]
            
            # Add neurons for each layer
            for _ in range(self.max_layers):
                if _ < num_layers:
                    chromosome.append(random.randint(8, self.max_neurons))
                else:
                    chromosome.append(0)  # Unused layers
            
            # Add learning rate and activation indices
            chromosome.append(random.randint(0, len(self.learning_rates) - 1))
            chromosome.append(random.randint(0, len(self.activations) - 1))
            
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
                if i == 0:  # num_layers
                    mutated[i] = random.randint(1, self.max_layers)
                elif i < self.max_layers + 1:  # neurons
                    if i <= mutated[0]:  # Only mutate active layers
                        mutated[i] = random.randint(8, self.max_neurons)
                elif i == len(mutated) - 2:  # learning rate index
                    mutated[i] = random.randint(0, len(self.learning_rates) - 1)
                elif i == len(mutated) - 1:  # activation index
                    mutated[i] = random.randint(0, len(self.activations) - 1)
        
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
                loss = self.train_model(X_scaled, y_scaled, chromosome)
                fitness = self.fitness_function(loss)
                fitness_scores.append(fitness)
                
                # Track best chromosome
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = chromosome.copy()
            
            self.fitness_scores = fitness_scores
            avg_fitness = np.mean(fitness_scores)
            max_fitness = max(fitness_scores)
            
            print(f"Average fitness: {avg_fitness:.4f}, Best fitness: {max_fitness:.4f}")
            print(f"Best chromosome so far: {self.best_chromosome}")
            
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
        print(f"Fitness: {self.best_fitness:.4f}")
        
        # Decode and print best solution
        num_layers, neurons, lr, activation = self.decode_chromosome(self.best_chromosome)
        print(f"Best NN architecture:")
        print(f"  Layers: {num_layers}")
        print(f"  Neurons per layer: {neurons[:num_layers]}")
        print(f"  Learning rate: {lr}")
        print(f"  Activation: {activation}")
        
        return self.best_chromosome


def main():
    """Main function to run the genetic algorithm"""
    print("=== Genetic Algorithm for Neural Network Optimization ===")
    
    # Initialize GA
    ga = GeneticAlgorithm(
        generation_size=20,
        crossover_p=0.8,
        mutation_p=0.2,
        max_layers=3,
        max_neurons=64,
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
    
    # Train final model with best chromosome
    print("\nTraining final model with best parameters...")
    final_loss = ga.train_model(X_selected, y_scaled, best_chromosome)
    print(f"Final model loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()