# Genetic Algorithm Neural Network Optimizer

A complete implementation of a Genetic Algorithm for optimizing neural network hyperparameters for time series prediction (stock prices).

## 🚀 Features

- **Complete Genetic Algorithm Implementation**: Selection, crossover, mutation, and elitism
- **Neural Network Optimization**: Optimizes LSTM/GRU architectures and hyperparameters
- **Time Series Support**: Built-in sequence preparation for stock price prediction
- **Multiple Layer Types**: LSTM, GRU, Bidirectional LSTM, Bidirectional GRU
- **Comprehensive Tracking**: Evolution history and fitness tracking
- **Visualization**: Automatic plotting of results and evolution progress
- **Flexible Data Input**: Works with custom CSV files or generates sample data

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🔧 What Gets Optimized

The genetic algorithm optimizes the following neural network hyperparameters:

- **Layer Type**: LSTM, GRU, Bidirectional LSTM, Bidirectional GRU
- **Number of Layers**: 1-3 recurrent layers
- **Units per Layer**: 32-128 neurons per layer
- **Dropout Rate**: 0.1-0.5
- **Learning Rate**: 0.0001-0.01
- **Batch Size**: 16, 32, or 64
- **Optimizer**: Adam or RMSprop

## 🎯 Quick Start

### Option 1: Use Sample Data (Default)

```python
from genetic_neural_optimizer import GeneticNeuralOptimizer

# Initialize the optimizer
ga_optimizer = GeneticNeuralOptimizer(
    population_size=10,
    crossover_rate=0.8,
    mutation_rate=0.15,
    generations=5,
    sequence_length=30
)

# Load sample data and run optimization
ga_optimizer.load_and_prepare_data()
best_individual = ga_optimizer.run_evolution()

# Train the best model
best_model, history = ga_optimizer.train_best_model(epochs=50)

# Make predictions and plot results
predictions = ga_optimizer.make_predictions(best_model)
```

### Option 2: Use Your Own Data

```python
# Load your own CSV file
success = ga_optimizer.load_and_prepare_data(
    file_path="your_stock_data.csv",
    features=['Close', 'Volume', 'High', 'Low'],  # Input features
    target=['Close']  # What to predict
)
```

### Option 3: Run the Example Script

```bash
python example_usage.py
```

## 📊 How It Works

### 1. **Chromosome Representation**
Each individual in the population represents a neural network configuration:
```python
chromosome = {
    'layer_type': 'LSTM',
    'num_layers': 2,
    'units': [64, 32, 16],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam'
}
```

### 2. **Fitness Evaluation**
For each chromosome:
- Build neural network with specified parameters
- Train on training data (with early stopping)
- Evaluate on validation data
- Return negative validation loss as fitness

### 3. **Genetic Operations**
- **Selection**: Tournament selection (best of 3 random individuals)
- **Crossover**: Parameter swapping between parents
- **Mutation**: Random parameter modification
- **Elitism**: Best individual always survives

### 4. **Evolution Process**
```
Generation 1: Random population → Evaluate fitness → Select parents
Generation 2: Crossover → Mutation → Evaluate → Select...
...
Final: Return best neural network configuration
```

## 📈 Output

The optimizer generates:

1. **Evolution History Plot**: Shows best and average fitness over generations
2. **Prediction Plots**: Actual vs predicted values for training and validation
3. **Best Configuration**: Optimal hyperparameters found
4. **Trained Model**: Ready-to-use TensorFlow model

## ⚙️ Configuration Options

### GeneticNeuralOptimizer Parameters

```python
GeneticNeuralOptimizer(
    population_size=20,    # Number of individuals per generation
    crossover_rate=0.8,    # Probability of crossover (0.0-1.0)
    mutation_rate=0.1,     # Probability of mutation (0.0-1.0)
    generations=10,        # Number of generations to evolve
    sequence_length=60     # Days of history for prediction
)
```

### Performance vs Speed Trade-offs

**For Faster Results (Demo)**:
```python
population_size=8, generations=3, epochs=20
```

**For Better Results (Production)**:
```python
population_size=20, generations=15, epochs=100
```

## 📁 File Structure

```
├── genetic_neural_optimizer.py    # Main GA implementation
├── example_usage.py              # Usage examples
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── evolution_history.png         # Generated: Evolution plot
└── predictions.png               # Generated: Prediction plot
```

## 🔍 Example Results

After running the genetic algorithm, you'll see:

```
Generation 1/5
Individual 1: Fitness = -0.012345
Individual 2: Fitness = -0.015678
...
Best fitness: -0.008901, Average fitness: -0.013456

Best configuration found: {
    'layer_type': 'Bidirectional_LSTM',
    'num_layers': 2,
    'units': [96, 64, 32],
    'dropout_rate': 0.25,
    'learning_rate': 0.0023,
    'batch_size': 32,
    'optimizer': 'adam'
}
```

## 💡 Tips for Best Results

1. **Start Small**: Use small population and generations for testing
2. **Multiple Runs**: GA is stochastic - run multiple times for best results
3. **Data Quality**: Ensure your CSV has consistent date/price columns
4. **Memory Management**: Larger populations require more GPU/CPU memory
5. **Early Stopping**: Built-in to prevent overfitting during evolution

## 🐛 Troubleshooting

**Memory Issues**: Reduce `population_size` or `sequence_length`
**Slow Execution**: Reduce `generations` or `epochs` per evaluation
**Poor Results**: Increase `population_size` and `generations`
**Data Errors**: Ensure CSV has required columns (Close, Volume, etc.)

## 🔄 Advanced Usage

### Custom Fitness Function

You can modify the `_evaluate_fitness` method to use different metrics:

```python
def _evaluate_fitness(self, chromosome):
    # Custom implementation
    # Return higher values for better individuals
    pass
```

### Different Network Architectures

Extend the `_create_model` method to support:
- Convolutional layers
- Attention mechanisms
- Different optimizers
- Custom loss functions

## 📚 References

- Genetic Algorithms: Holland, J.H. (1992)
- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Time Series Prediction: Multiple approaches and methodologies

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

---

**Note**: This is a computationally intensive process. Each generation requires training multiple neural networks. Start with small parameters and scale up as needed.