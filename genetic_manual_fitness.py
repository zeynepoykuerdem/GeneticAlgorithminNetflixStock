import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ManualNeuralNetwork:
    """Simple neural network implemented from scratch"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 learning_rate: float = 0.01, activation: str = 'tanh'):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('tanh', 'sigmoid', 'relu')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Build layer sizes
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(1.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'tanh':
            return np.tanh(np.clip(x, -10, 10))
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x  # linear
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function"""
        if self.activation == 'tanh':
            tanh_x = np.tanh(np.clip(x, -10, 10))
            return 1 - tanh_x**2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -10, 10)))
            return sig * (1 - sig)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            return np.ones_like(x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward propagation"""
        activations = [X]
        z_values = []
        current_input = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, w) + b
            z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer (linear)
                current_input = z
            else:  # Hidden layers
                current_input = self._activation_function(z)
            
            activations.append(current_input)
        
        return current_input, activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                z_values: List[np.ndarray]) -> float:
        """Backward propagation and weight update"""
        m = X.shape[0]
        
        # Calculate loss (MSE)
        output = activations[-1]
        loss = np.mean((output - y)**2)
        
        # Initialize error for output layer
        delta = 2 * (output - y) / m
        
        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Calculate error for previous layer (if not input layer)
            if i > 0:
                # Propagate error backward
                delta = np.dot(delta, self.weights[i].T)
                # Apply activation derivative (for hidden layers)
                delta *= self._activation_derivative(z_values[i-1])
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = False) -> List[float]:
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            try:
                output, activations, z_values = self.forward(X)
                loss = self.backward(X, y, activations, z_values)
                losses.append(loss)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}")
                    
                # Check for convergence issues
                if np.isnan(loss) or np.isinf(loss):
                    break
                    
            except Exception as e:
                print(f"Training error at epoch {epoch}: {e}")
                break
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _, _ = self.forward(X)
        return output


class GeneticManualOptimizer:
    """Genetic Algorithm with manually implemented fitness function"""
    
    def __init__(self, population_size: int = 20, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.1, generations: int = 10, sequence_length: int = 30):
        """
        Initialize the Genetic Algorithm with manual fitness evaluation
        
        Args:
            population_size: Number of individuals in each generation
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            generations: Number of generations to evolve
            sequence_length: Length of input sequences for time series
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
        
        # Neural network constraints
        self.min_layers = 1
        self.max_layers = 4
        self.min_neurons = 5
        self.max_neurons = 50
        self.min_lr = 0.001
        self.max_lr = 0.1
        self.min_epochs = 20
        self.max_epochs = 100
    
    def load_and_prepare_data(self, file_path: str = None, features: List[str] = None) -> bool:
        """Load and prepare time series data for training"""
        if file_path is None:
            # Generate sample stock data
            print("Generating sample stock data...")
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic stock price with trend and noise
            base_price = 100
            trend = 0.0002
            prices = []
            price = base_price
            
            for i in range(len(dates)):
                daily_return = np.random.normal(trend, 0.02)
                price *= (1 + daily_return)
                prices.append(price)
            
            # Create features
            data = pd.DataFrame({
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates)),
                'High': np.array(prices) * (1 + np.random.uniform(0, 0.05, len(dates))),
                'Low': np.array(prices) * (1 - np.random.uniform(0, 0.05, len(dates))),
            })
            
            # Add technical indicators
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['Returns'] = data['Close'].pct_change()
            data.dropna(inplace=True)
            
        else:
            try:
                data = pd.read_csv(file_path)
                print(f"Loaded data from {file_path}")
            except Exception as e:
                print(f"Error loading data: {e}")
                return False
        
        print(f"Data shape: {data.shape}")
        
        # Use available features
        available_features = ['Close', 'Volume', 'High', 'Low', 'MA_5', 'MA_20', 'Returns']
        features_to_use = [f for f in available_features if f in data.columns]
        
        if not features_to_use:
            features_to_use = ['Close']
        
        print(f"Using features: {features_to_use}")
        
        # Manual normalization (without sklearn)
        X_data = data[features_to_use].values
        y_data = data['Close'].values.reshape(-1, 1)
        
        # Min-max normalization
        self.X_min = np.min(X_data, axis=0)
        self.X_max = np.max(X_data, axis=0)
        self.y_min = np.min(y_data)
        self.y_max = np.max(y_data)
        
        X_scaled = (X_data - self.X_min) / (self.X_max - self.X_min + 1e-8)
        y_scaled = (y_data - self.y_min) / (self.y_max - self.y_min + 1e-8)
        
        # Create sequences for time series
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled.flatten())
        
        # Train/validation split (80/20)
        split_idx = int(len(X_sequences) * 0.8)
        self.X_train, self.X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        self.y_train, self.y_val = y_sequences[:split_idx], y_sequences[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}, {self.y_val.shape}")
        
        return True
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            # Flatten the sequence to create feature vector
            X_seq.append(X[i-self.sequence_length:i].flatten())
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq).reshape(-1, 1)
    
    def _create_chromosome(self) -> Dict[str, Any]:
        """Create a random chromosome representing neural network hyperparameters"""
        num_layers = random.randint(self.min_layers, self.max_layers)
        
        # Calculate input size based on sequence length and features
        input_size = self.X_train.shape[1] if self.X_train is not None else 30
        
        chromosome = {
            'num_layers': num_layers,
            'layer_sizes': [random.randint(self.min_neurons, self.max_neurons) for _ in range(num_layers)],
            'learning_rate': random.uniform(self.min_lr, self.max_lr),
            'epochs': random.randint(self.min_epochs, self.max_epochs),
            'activation': random.choice(['tanh', 'sigmoid', 'relu']),
            'input_size': input_size,
            'output_size': 1  # Single output for regression
        }
        
        return chromosome
    
    def _manual_fitness_evaluation(self, chromosome: Dict[str, Any]) -> float:
        """
        Manually implemented fitness function without using pre-built models
        
        Args:
            chromosome: Neural network configuration
            
        Returns:
            fitness: Negative mean squared error (higher is better)
        """
        try:
            # Create and train neural network manually
            nn = ManualNeuralNetwork(
                input_size=chromosome['input_size'],
                hidden_sizes=chromosome['layer_sizes'],
                output_size=chromosome['output_size'],
                learning_rate=chromosome['learning_rate'],
                activation=chromosome['activation']
            )
            
            # Train the network
            train_losses = nn.train(
                self.X_train, 
                self.y_train, 
                epochs=chromosome['epochs'],
                verbose=False
            )
            
            # Evaluate on validation set
            val_predictions = nn.predict(self.X_val)
            
            # Manual MSE calculation
            mse = np.mean((val_predictions - self.y_val)**2)
            
            # Additional penalties for complexity (regularization)
            complexity_penalty = sum(chromosome['layer_sizes']) * 0.0001
            training_stability = np.std(train_losses[-10:]) if len(train_losses) >= 10 else 0
            
            # Fitness is negative MSE minus penalties (higher is better)
            fitness = -mse - complexity_penalty - training_stability
            
            return fitness
            
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return float('-inf')
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for choosing parents"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parent chromosomes"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Crossover parameters
        for key in ['learning_rate', 'epochs']:
            if random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]
        
        # Crossover activation function
        if random.random() < 0.5:
            child1['activation'], child2['activation'] = child2['activation'], child1['activation']
        
        # Crossover layer sizes (if same number of layers)
        if parent1['num_layers'] == parent2['num_layers']:
            for i in range(parent1['num_layers']):
                if random.random() < 0.5:
                    child1['layer_sizes'][i], child2['layer_sizes'][i] = \
                        child2['layer_sizes'][i], child1['layer_sizes'][i]
        
        return child1, child2
    
    def _mutate(self, chromosome: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mutation on a chromosome"""
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['layers', 'learning_rate', 'epochs', 'activation'])
            
            if mutation_type == 'layers':
                # Mutate layer sizes
                if mutated['layer_sizes']:
                    idx = random.randint(0, len(mutated['layer_sizes']) - 1)
                    mutated['layer_sizes'][idx] = random.randint(self.min_neurons, self.max_neurons)
                
            elif mutation_type == 'learning_rate':
                mutated['learning_rate'] = random.uniform(self.min_lr, self.max_lr)
                
            elif mutation_type == 'epochs':
                mutated['epochs'] = random.randint(self.min_epochs, self.max_epochs)
                
            elif mutation_type == 'activation':
                mutated['activation'] = random.choice(['tanh', 'sigmoid', 'relu'])
        
        return mutated
    
    def run_evolution(self) -> Dict[str, Any]:
        """Run the genetic algorithm evolution process"""
        if self.X_train is None:
            print("Error: No data loaded. Please call load_and_prepare_data() first.")
            return None
        
        print(f"Starting manual fitness evolution with {self.population_size} individuals for {self.generations} generations")
        print("Note: Using manually implemented neural networks and fitness evaluation")
        
        # Initialize population
        population = [self._create_chromosome() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate fitness for all individuals using manual implementation
            fitness_scores = []
            for i, chromosome in enumerate(population):
                fitness = self._manual_fitness_evaluation(chromosome)
                fitness_scores.append(fitness)
                
                # Print individual info
                layers_info = f"[{chromosome['input_size']}→" + "→".join(map(str, chromosome['layer_sizes'])) + f"→{chromosome['output_size']}]"
                print(f"Individual {i+1}: Layers {layers_info}, LR={chromosome['learning_rate']:.4f}, "
                      f"Epochs={chromosome['epochs']}, Activation={chromosome['activation']}, Fitness={fitness:.6f}")
            
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
        plt.title('Manual Fitness Evolution')
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
        plt.savefig('manual_evolution_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_best_model(self) -> ManualNeuralNetwork:
        """Train the best model found by the genetic algorithm"""
        if self.best_individual is None:
            print("Error: No best individual found. Please run evolution first.")
            return None
        
        print("Training best model with manual implementation...")
        print(f"Best parameters: {self.best_individual}")
        
        # Create the best neural network
        best_nn = ManualNeuralNetwork(
            input_size=self.best_individual['input_size'],
            hidden_sizes=self.best_individual['layer_sizes'],
            output_size=self.best_individual['output_size'],
            learning_rate=self.best_individual['learning_rate'],
            activation=self.best_individual['activation']
        )
        
        # Train the network
        print("Training...")
        train_losses = best_nn.train(
            self.X_train, 
            self.y_train, 
            epochs=self.best_individual['epochs'],
            verbose=True
        )
        
        # Evaluate manually
        train_pred = best_nn.predict(self.X_train)
        val_pred = best_nn.predict(self.X_val)
        
        # Manual MSE and MAE calculation
        train_mse = np.mean((train_pred - self.y_train)**2)
        val_mse = np.mean((val_pred - self.y_val)**2)
        train_mae = np.mean(np.abs(train_pred - self.y_train))
        val_mae = np.mean(np.abs(val_pred - self.y_val))
        
        # Manual R² calculation
        train_ss_res = np.sum((self.y_train - train_pred) ** 2)
        train_ss_tot = np.sum((self.y_train - np.mean(self.y_train)) ** 2)
        train_r2 = 1 - (train_ss_res / (train_ss_tot + 1e-8))
        
        val_ss_res = np.sum((self.y_val - val_pred) ** 2)
        val_ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        val_r2 = 1 - (val_ss_res / (val_ss_tot + 1e-8))
        
        print(f"Manual Training Metrics - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.4f}")
        print(f"Manual Validation Metrics - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.4f}")
        
        return best_nn
    
    def make_predictions(self, model: ManualNeuralNetwork, plot_results: bool = True):
        """Make predictions using the trained model"""
        # Make predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        # Denormalize predictions manually
        train_pred_denorm = train_pred * (self.y_max - self.y_min) + self.y_min
        val_pred_denorm = val_pred * (self.y_max - self.y_min) + self.y_min
        y_train_denorm = self.y_train * (self.y_max - self.y_min) + self.y_min
        y_val_denorm = self.y_val * (self.y_max - self.y_min) + self.y_min
        
        if plot_results:
            plt.figure(figsize=(15, 8))
            
            # Plot training predictions
            plt.subplot(2, 1, 1)
            plt.scatter(y_train_denorm[:200], train_pred_denorm[:200], alpha=0.6, label='Predictions')
            plt.plot([y_train_denorm[:200].min(), y_train_denorm[:200].max()], 
                    [y_train_denorm[:200].min(), y_train_denorm[:200].max()], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Manual NN Training Predictions (First 200 samples)')
            plt.legend()
            plt.grid(True)
            
            # Plot validation predictions
            plt.subplot(2, 1, 2)
            plt.scatter(y_val_denorm, val_pred_denorm, alpha=0.6, label='Predictions')
            plt.plot([y_val_denorm.min(), y_val_denorm.max()], 
                    [y_val_denorm.min(), y_val_denorm.max()], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Manual NN Validation Predictions')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('manual_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return (y_train_denorm, train_pred_denorm, y_val_denorm, val_pred_denorm)


# Example usage
if __name__ == "__main__":
    print("=== Genetic Algorithm with Manual Fitness Implementation ===\n")
    
    # Initialize the genetic optimizer
    ga_optimizer = GeneticManualOptimizer(
        population_size=6,   # Smaller for faster demonstration
        crossover_rate=0.8,
        mutation_rate=0.15,
        generations=3,       # Fewer generations for demo
        sequence_length=15   # Shorter sequences
    )
    
    # Load and prepare data
    success = ga_optimizer.load_and_prepare_data()
    
    if success:
        print("Running evolution with manually implemented fitness function...")
        # Run evolution
        best_individual = ga_optimizer.run_evolution()
        
        # Plot evolution history
        ga_optimizer.plot_evolution_history()
        
        # Train the best model
        best_model = ga_optimizer.train_best_model()
        
        # Make predictions and plot results
        predictions = ga_optimizer.make_predictions(best_model)
        
        print("\nGenetic Algorithm with Manual Implementation Complete!")
        print(f"Best configuration found: {best_individual}")
    else:
        print("Failed to load data. Exiting...")