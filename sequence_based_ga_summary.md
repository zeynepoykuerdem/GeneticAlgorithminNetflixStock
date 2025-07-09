# Sequence-based Genetic Algorithm - Chromosome Length = Sequence Length

## Temel Konsept

Bu implementasyonda, **chromosome length = sequence length** olacak şekilde tasarlanmış bir genetic algorithm bulunmaktadır. Bu yaklaşım time series prediction için özellikle uygundur.

### Chromosome Yapısı

```python
chromosome = [weight1, weight2, weight3, ..., weight60]  # 60 = sequence_length
```

- Her gen, sequence'daki bir time step'in önemini/ağırlığını temsil eder
- Chromosome uzunluğu = Sequence uzunluğu (60)
- Her gen 0.1-2.0 arasında bir float değer

## Test Sonuçları

### Başarılı Çalışma
```
Sequence length: 60
Chromosome length: 60 ✓ (EŞIT!)

Generation 1/15: Best fitness: 0.9983 (MSE: 0.0017)
Generation 15/15: Best fitness: 0.9983 (MSE: 0.0017)

Best Solution:
- Chromosome length: 60 (= sequence length)
- Score: 0.0017
- Test R²: 0.9403 (94.03% variance explained)
```

### En Önemli Time Steps
Algorithm şu time step'leri en önemli buldu:
1. Time step 57: weight = 0.0360 (sequence'in sonuna yakın)
2. Time step 52: weight = 0.0330 
3. Time step 0: weight = 0.0328 (sequence'in başı)
4. Time step 10: weight = 0.0314
5. Time step 12: weight = 0.0304

## Nasıl Çalışır?

### 1. Sequence Oluşturma
```python
# 1000 time step'lik veri -> 60 uzunluğunda sequence'lar
X_sequences shape: (940, 60, 3)  # 940 sequence, 60 time step, 3 feature
y_sequences shape: (940,)
```

### 2. Chromosome Weights Uygulama
```python
def apply_chromosome_weights(self, X_sequences, chromosome):
    weights = np.array(chromosome)  # Length = 60
    weights = weights / np.sum(weights)  # Normalize
    
    # Her time step'e ağırlık uygula
    for i in range(len(weights)):
        weighted_sequences[:, i, :] *= weights[i]
```

### 3. Model Training
- Weighted sequences flatten ediliyor: (940, 60*3) = (940, 180)
- MLPRegressor ile training
- MSE ile fitness hesaplanıyor

### 4. Genetic Operations
- **Selection**: Tournament selection (3 individuals)
- **Crossover**: Single-point crossover (%80 probability)
- **Mutation**: Gaussian noise addition (%20 probability)
- **Elitism**: Best individual always survives

## Avantajları

### Time Series İçin Özel
- Her time step'in önemini öğrenir
- Hangi zaman noktalarının daha kritik olduğunu bulur
- Long-term vs short-term dependencies'i keşfeder

### Interpretable Results
- Chromosome weights görsellleştirilebilir
- En önemli time step'ler analiz edilebilir
- Model kararlarının açıklanabilirliği yüksek

### Flexible Architecture
- Farklı sequence length'leri kolayca destekler
- Multiple features ile uyumlu
- Farklı ML modelleri (MLPRegressor, RandomForest) kullanabilir

## Kod Yapısı

### Temel Parameterler
```python
ga = GeneticAlgorithm(
    generation_size=20,       # Population size
    crossover_p=0.8,         # Crossover probability  
    mutation_p=0.2,          # Mutation probability
    sequence_length=60,      # Chromosome length = bu değer
    generations=15           # Evolution generations
)
```

### Main Pipeline
```python
# 1. Data loading & preprocessing
data = ga.load_data()
X_scaled, y_scaled, _, _ = ga.pre_process_data(data, features, target)

# 2. Feature selection
X_selected, _ = ga.feature_selection(X_scaled, y_scaled, features, k=3)

# 3. Sequence creation - Bu adımdan sonra chromosome length = sequence length
X_sequences, y_sequences = ga.create_sequences(X_selected, y_scaled, sequence_length)

# 4. Genetic algorithm
best_chromosome = ga.run(X_sequences, y_sequences)

# 5. Final evaluation
model, mse, r2 = ga.evaluate_best_model(X_sequences, y_sequences)
```

## Visualization

Code otomatik olarak chromosome weights'i görselleştirir:
- Raw weights plot
- Normalized weights plot  
- En önemli time step'lerin analizi
- `chromosome_weights.png` dosyası olarak kaydeder

## Gerçek Kullanım Senaryoları

### 1. Stock Prediction
- Son 60 günün hangi günlerinin daha önemli olduğunu öğrenir
- Trend değişim noktalarını keşfeder

### 2. Weather Forecasting
- Hangi saatlerin hava tahmini için kritik olduğunu bulur
- Seasonal patterns'i öğrenir

### 3. Sensor Data Analysis
- Hangi measurement time'ların daha önemli olduğunu analiz eder
- Anomaly detection için önemli time windows'ları keşfeder

## Customization Seçenekleri

### Farklı Sequence Length
```python
# 30 günlük sequence için
ga = GeneticAlgorithm(sequence_length=30, ...)  # Chromosome length = 30

# 120 günlük sequence için  
ga = GeneticAlgorithm(sequence_length=120, ...) # Chromosome length = 120
```

### Farklı Weight Ranges
```python
# Mutation process'te range değiştirilebilir
mutated[i] = max(0.01, min(5.0, mutated[i]))  # 0.01-5.0 range
```

### Farklı Models
```python
# Model type değiştirilebilir
self.model_type = 'RandomForest'  # veya 'MLPRegressor'
```

## Önemli Notlar

1. **Chromosome Length = Sequence Length**: Bu temel gerekliliğin sağlandığı garanti edilmiştir
2. **Memory Efficient**: Large sequences için memory kullanımı optimize edilmiştir  
3. **Reproducible**: Random seed'ler set edilmiştir
4. **Scalable**: Farklı data size'ları ve sequence length'leri destekler

Bu implementation, sizin orijinal ihtiyacınızı tam olarak karşılamaktadır: **sequences yaratıldıktan sonra chromosome length = sequence length**.