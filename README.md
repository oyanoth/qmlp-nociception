# A Quantum Multi-Layer Perceptron (QMLP) for Nociception Prediction

Implementation of QMLP architecture for real-time intraoperative nociception assessment using quantum machine learning.

## Overview

This repository contains the complete implementation of a Quantum Multi-Layer Perceptron (QMLP) in nociception prediction during surgical procedures. The model uses quantum circuit learning with data re-uploading to process physiological signals and predict pain responses.

**Key Features:**
- Hybrid quantum-classical neural network architecture
- Multiple circuit topologies (Circular, Linear, Pairwise)
- Parameter shift rule for exact gradient computation
- Parallel training with joblib
- Real-time inference capability (0.33 ms/sample)
- Entanglement analysis tools

## Requirements

### Software Dependencies
Our models were implemnted using the following versions

```bash
numpy        installed: 2.4.0     
qulacs       installed: 0.6.12    
torch        installed: 2.9.1+cpu 
joblib       installed: 1.5.3     
psutil       installed: 7.1.3      
tqdm         installed: 4.67.1     
sklearn      installed: 1.8.0    
```

### Hardware Requirements

- **Recommended:** 32 GB RAM, 16 CPU cores
- **Tested on:** AMD EPYC 7B13 (16 logical cores, 31.35 GB RAM)

## Installation

```bash
git clone https://github.com/oyanoth/qmlp-nociception.git
cd qmlp-nociception
pip install -r requirements.txt
```

## Quick Start

### Basic Training Example

```python
import numpy as np
from model import QMLP
from trainer import AdamTrainer
from utils import set_reproducibility

# Set reproducibility
set_reproducibility(seed=42)

# Load your data (X: features, y: targets)
X_train = np.load('data/X_train.npy')  # Shape: (N, n_qubits)
y_train = np.load('data/y_train.npy')  # Shape: (N,)
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')

# Initialize model
model = QMLP(
    n_qubits=4,
    topology="Circular",
    layers=2,    # This is the number of re-uploading layers, layers=2 means the model has 3 hidden layers
    use_scaler=True, # This can be set based on experiments, change to None if not needed and remove scaling_angle
    scaling_angle=np.pi, # If scaling is used, orignal parameter is recommended to be in range [0, 1]
    activation="mapping", # mapping linearly maps expectation value to the range [0, 1], modify the original function if activation is not needed
    loss="mse"
)

# Initialize trainer
trainer = AdamTrainer(model, lr=0.1, betas=(0.9, 0.999))

# Train
best_params, peak_memory = trainer.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=100,
    batch_size=32,
    n_jobs=-1,  # Use all available cores
    patience=10,
    validation_freq=1
)

# Predict
predictions = model.predict(X_val, best_params)
```

### Entanglement Analysis

```python
from entanglement_analyzer import EntanglementAnalyzer

analyzer = EntanglementAnalyzer(
    n_qubits=4,
    topology="Circular",
    layers=2, 
    use_scaler=True,
    scaling_angle=np.pi
)

results = analyzer.analyze_dataset(
    X=X_test,
    params=best_params,
    n_samples=1000,
    verbose=True
)

print(f"Mean normalized entanglement: {results['mean_normalized_max']:.4f}")
print(f"Haar entropy bound: {results['S_haar_max']:.4f}")
```


## Architecture Details

### QMLP Model (`model.py`)

The quantum circuit implements a data re-uploading architecture:

1. **S-layer (Encoding):** Hadamard gates followed by RY rotations with input features
2. **W-layer (Variational):** Parameterized RY rotations with topology-dependent CNOTs
3. **Re-uploading:** Alternating W and S layers for L iterations
4. **Measurement:** Z⊗Z⊗...⊗Z observable on all qubits

**Topologies:**
- **Circular:** Full ring connectivity with CNOTs (i → i+1, n-1 → 0)
- **Linear:** Chain connectivity (i → i+1)
- **Pairwise:** Nearest-neighbor pairs (0→1, 2→3, ...)


## Project Structure

```
qmlp-nociception/
├── model.py                    # QMLP quantum circuit implementation
├── trainer.py                  # Adam optimizer with parallel processing
├── utils.py                    # Reproducibility and utility functions
├── entanglement_analyzer.py    # Entanglement entropy analysis tools
├── requirements.txt            # Python dependencies
├── README.md                   # This file
```



## Reproducibility

All experiments use a fixed random seed (42) for full reproducibility:

```python
from utils import set_reproducibility
set_reproducibility(seed=42)
```

This configures:
- Python's `random` module
- NumPy random state
- PyTorch (CPU and CUDA)
- Optuna samplers (if installed)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{To be modifed,
  title={Quantum Multi-Layer Perceptron for Nociception Monitoring},
  author={To be modifed},
  journal={To be modifed},
  year={To be modifed}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
