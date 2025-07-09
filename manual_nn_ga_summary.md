# Manual Neural Network + Genetic Algorithm Implementation

## ✅ Başarılı Implementation!

Bu version'da **manual neural network** implementation'ı ve **chromosome length = sequence length** gerekliliğinin her ikisi de başarıyla implement edilmiştir.

## Test Sonuçları

### Başarılı Çalışma:
```
Sequence length: 60
Chromosome length: 60 ✓ (TAMAMEN EŞIT!)
Input to NN: 60 * 3 = 180 neurons

Generation 1/10: Best fitness: 0.8730 (MSE: 0.1455)
Generation 10/10: Best fitness: 0.8563 (MSE: 0.1399)

Manual Neural Network Training:
Epoch 0, Loss: 0.270324
Epoch 80, Loss: 0.199646

Final Results:
- Chromosome length: 60 (= sequence length) ✓
- Test MSE: 0.1615
- Manual NN çalışıyor ✓
```

## Manual Neural Network Özellikleri

### 1. **Tam Manual Implementation**
```python
class ManualNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size=1, learning_rate=0.001):
        # Xavier weight initialization
        # Manual forward pass
        # Manual backpropagation
        # Manual weight updates
```

### 2. **Neural Network Yapısı**
- **Input Layer**: 180 neurons (60 time steps × 3 features)
- **Hidden Layer 1**: 64 neurons + ReLU activation
- **Hidden Layer 2**: 32 neurons + ReLU activation  
- **Output Layer**: 1 neuron (regression)

### 3. **Manual Components**
- ✅ **Forward Pass**: Manuel matrix operations
- ✅ **ReLU Activation**: `np.maximum(0, x)`
- ✅ **Backpropagation**: Manual gradient calculation
- ✅ **Weight Updates**: Manual gradient descent
- ✅ **Xavier Initialization**: `w * sqrt(2.0 / input_size)`

## Chromosome Yapısı

### Her Gen = Time Step Weight
```python
chromosome = [w1, w2, w3, ..., w60]  # Length = 60 (sequence length)

# Her weight, o time step'in önemini belirler:
weighted_sequences[:, i, :] *= chromosome[i]  # i'nci time step'e ağırlık
```

### En Önemli Time Steps
Algorithm şunları buldu:
1. **Time step 16**: weight = 0.0380 (en önemli)
2. **Time step 0**: weight = 0.0375 (sequence başı)
3. **Time step 48**: weight = 0.0368 (sequence sonu yakın)
4. **Time step 33**: weight = 0.0340 (ortalar)
5. **Time step 8**: weight = 0.0323

## Genetic Algorithm Flow

### 1. **Initialization**
```python
# 15 chromosomes, each length 60
population = [[w1, w2, ..., w60] for _ in range(15)]
```

### 2. **Fitness Evaluation**
```python
for chromosome in population:
    # Apply weights to sequences
    weighted_data = apply_chromosome_weights(sequences, chromosome)
    
    # Train manual neural network
    model = ManualNeuralNetwork(input_size=180, hidden_layers=[64, 32])
    model.fit(weighted_data, targets, epochs=50)
    
    # Calculate fitness
    mse = evaluate(model)
    fitness = 1 / (1 + mse)
```

### 3. **Evolution Operations**
- **Tournament Selection**: 3-individual tournament
- **Single-point Crossover**: 80% probability
- **Gaussian Mutation**: 20% probability  
- **Elitism**: Best individual survives

## Avantajları

### 1. **Tam Manual Control**
- No pre-built models (MLPRegressor, etc.)
- Every operation manually implemented
- Full understanding of neural network process

### 2. **Time Series Specific**
- Chromosome length = sequence length
- Each gene controls time step importance
- Learns temporal patterns automatically

### 3. **Interpretable Results**
- Can see which time steps matter most
- Weight visualization available
- Clear understanding of model decisions

### 4. **Flexible Architecture**
- Easy to modify NN architecture
- Configurable hidden layers
- Adjustable learning parameters

## Manual Neural Network Details

### Forward Pass Implementation:
```python
def forward(self, X):
    current_input = X
    # Hidden layers with ReLU
    for i in range(len(self.weights) - 1):
        z = np.dot(current_input, self.weights[i]) + self.biases[i]
        current_input = self.relu(z)  # ReLU activation
    
    # Output layer (linear)
    output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
    return output
```

### Backpropagation Implementation:
```python
def backward(self, X, y, output):
    # Output layer error
    dZ = output - y.reshape(-1, 1)
    
    # Backpropagate through all layers
    for i in reversed(range(len(self.weights))):
        dW_i = np.dot(self.activations[i].T, dZ) / m
        db_i = np.sum(dZ, axis=0, keepdims=True) / m
        
        if i > 0:  # Hidden layers
            dA = np.dot(dZ, self.weights[i].T)
            dZ = dA * self.relu_derivative(self.z_values[i-1])
```

## Usage Example

```python
# Initialize with manual NN
ga = GeneticAlgorithm(
    generation_size=15,
    crossover_p=0.8,
    mutation_p=0.2,
    sequence_length=60,        # Chromosome length = this
    hidden_layers=[64, 32],    # Manual NN architecture
    learning_rate=0.001,       # Manual NN learning rate
    generations=10
)

# Create sequences
X_sequences, y_sequences = ga.create_sequences(data, target, sequence_length=60)

# Run GA with manual NN
best_chromosome = ga.run(X_sequences, y_sequences)
```

## Key Differences from Original

### ❌ Original Issues:
- Used pre-built models (TensorFlow/Keras)
- Had model type selection
- Incomplete implementations

### ✅ Fixed Version:
- **100% manual neural network**
- **No model type selection needed**
- **Complete implementations**
- **Chromosome length = sequence length guaranteed**

## Customization Options

### Different NN Architectures:
```python
hidden_layers=[128, 64, 32]    # 3 hidden layers
hidden_layers=[256]            # Single large layer
hidden_layers=[32, 16, 8]      # Decreasing sizes
```

### Different Activations:
```python
def tanh(self, x):
    return np.tanh(x)

def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
```

### Different Learning Rates:
```python
learning_rate=0.01     # Faster learning
learning_rate=0.0001   # Slower, more stable
```

Bu implementation tam olarak istediğiniz gibi: **Manual neural network + chromosome length = sequence length**. Artık hiçbir external model dependency'si yok ve her şey from scratch implement edilmiş!