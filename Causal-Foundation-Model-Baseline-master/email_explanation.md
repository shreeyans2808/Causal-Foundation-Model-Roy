## 📊 **What is predictions.npy?**

The predictions.npy file is the core output of the SEA Foundation Model for causal discovery. It contains probabilistic predictions for causal relationships between variables in your dataset.

## 🔍 **File Structure Overview**

The predictions.npy file follows this nested structure:
```
predictions.npy (NumPy array containing a dictionary)
├── [dataset_name] (e.g., 'hepar2', 'data')
    ├── 'pred': Probability predictions [shape: (1, N*(N-1))]
    ├── 'true': Ground truth (if available) [shape: (1, N*(N-1))]
    ├── 'time': Execution time
    ├── 'auc': Area Under ROC Curve
    └── 'prc': Precision-Recall Curve metrics
```

## 🎯 **Key Concepts**

### 1. **Why N*(N-1) predictions instead of N*N?**
- For N variables, a full adjacency matrix would be N×N = N² elements
- However, **diagonal elements are always excluded** (no self-loops: Variable A cannot cause itself)
- This leaves us with **N*(N-1) off-diagonal elements**

**Examples:**
- 70 variables → 70×70 matrix → 70*69 = 4,830 predictions
- 100 variables → 100×100 matrix → 100*99 = 9,900 predictions

### 2. **Probabilistic Output Format**
- Each prediction is a **probability between 0 and 1**
- Higher values indicate stronger evidence for a causal relationship
- Values are **NOT binary** - they represent confidence levels

### 3. **Matrix Reconstruction Process**
The 1D prediction array maps to a 2D adjacency matrix following this pattern:

```python
# Reconstruction Algorithm
adjacency_matrix = np.zeros((N, N))
prediction_index = 0

for i in range(N):          # Source variable (cause)
    for j in range(N):      # Target variable (effect)
        if i != j:          # Skip diagonal elements
            adjacency_matrix[i, j] = predictions_1d[prediction_index]
            prediction_index += 1
        # Diagonal remains 0 (no self-loops)
```

**Matrix Interpretation:**
- `adjacency_matrix[i, j]` = probability that Variable i causes Variable j
- Row i: All variables that Variable i might influence
- Column j: All variables that might influence Variable j

## 📈 **Practical Usage**

### **Step 1: Load Predictions**
```python
import numpy as np

# Load the file
data = np.load('predictions.npy', allow_pickle=True).item()

# Extract predictions (assuming dataset key is 'hepar2')
predictions_1d = np.array(data['hepar2']['pred'][0])
```

### **Step 2: Reconstruct Adjacency Matrix**
```python
import math

# Calculate number of variables
n_predictions = len(predictions_1d)
n_variables = int((1 + math.sqrt(1 + 4*n_predictions)) / 2)

# Reconstruct matrix
adjacency_matrix = np.zeros((n_variables, n_variables))
idx = 0
for i in range(n_variables):
    for j in range(n_variables):
        if i != j:
            adjacency_matrix[i, j] = predictions_1d[idx]
            idx += 1
```

### **Step 3: Apply Thresholds for Binary Decisions**
```python
# Convert probabilities to binary causal graph
threshold = 0.1  # Adjust based on your requirements
binary_graph = (adjacency_matrix > threshold).astype(int)

print(f"Detected {np.sum(binary_graph)} causal edges at threshold {threshold}")
```

## 🎛️ **Threshold Selection Guidelines**

- **Conservative (0.3-0.5)**: High precision, low recall - only very confident relationships
- **Moderate (0.1-0.2)**: Balanced precision/recall - good starting point
- **Liberal (0.01-0.05)**: High recall, low precision - captures more relationships but with noise

## 📊 **Evaluation Metrics**

When ground truth is available, you can evaluate using:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve

## 🔧 **Real-World Example**

Using our hepar2 dataset (hepatitis data):
- **Input**: 70 variables, 5,000 samples
- **Processing**: Model filtered to 65 variables
- **Output**: 4,160 predictions (65*64)
- **Matrix**: 65×65 adjacency matrix
- **Results**: Found medically meaningful relationships like "Cirrhosis → liver symptoms"

## ✨ **Key Advantages of This Format**

1. **Probabilistic**: Provides confidence levels, not just binary decisions
2. **Flexible**: Apply different thresholds for different use cases
3. **Efficient**: Stores only meaningful relationships (excludes diagonal)
4. **Standardized**: Consistent format across different datasets
5. **Interpretable**: Direct mapping to causal relationships

## 🚀 **Next Steps**

1. **Load your predictions.npy file**
2. **Reconstruct the adjacency matrix**
3. **Apply appropriate thresholds**
4. **Interpret the causal relationships**
5. **Validate with domain knowledge**
