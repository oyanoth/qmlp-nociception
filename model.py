import numpy as np
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import H, RY, CNOT

class QMLP:
    """
    Quantum Multi-Layer Perceptron (QMLP) implemented using Qulacs.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        topology (str): Circuit topology - "Circular", "Linear", or "Pairwise"
        layers (int): Number of variational layers
        use_scaler (bool): Whether to scale input data encoding
        scaling_angle (float): Scaling factor for angle embedding (default: π)
        activation (str): Activation function - "mapping", "relu", "sigmoid", "tanh", or None
        loss (str): Loss function - "mse" or "mae"
    """

    def __init__(self, n_qubits, topology="Circular", layers=1, 
                 use_scaler=False, scaling_angle=np.pi,
                 activation="mapping", loss="mse"):
        self.n_q = n_qubits
        self.topo = topology
        self.L = layers
        self.shift = np.pi / 2
        self.activation = activation.lower() if activation else None
        self.multiplier = scaling_angle if use_scaler else 1.0
        self.loss = loss.lower()
        
        if self.loss not in ["mse", "mae"]:
            raise ValueError(f"loss must be 'mse' or 'mae', got '{loss}'")
        
        if self.topo not in ["Circular", "Linear", "Pairwise"]:
            raise ValueError(f"topology must be 'Circular', 'Linear', or 'Pairwise', got '{topology}'")
    
    def _apply_activation(self, val):
        """Apply activation function to raw expectation value."""
        if self.activation == "mapping":
            return (val + 1) / 2
        elif self.activation == "relu":
            return max(0.0, val)
        elif self.activation == "sigmoid":
            # Clip to prevent overflow
            val_clipped = np.clip(val, -500, 500)
            return 1 / (1 + np.exp(-val_clipped))
        elif self.activation == "tanh":
            return np.tanh(val)
        return val

    def _activation_derivative(self, raw_val):
        """Compute the derivative of the activation function f'(x)."""
        if self.activation == "mapping":
            return 0.5
        elif self.activation == "relu":
            return 1.0 if raw_val > 0 else 0.0
        elif self.activation == "sigmoid":
            raw_val_clipped = np.clip(raw_val, -500, 500)
            s = 1 / (1 + np.exp(-raw_val_clipped))
            return s * (1 - s)
        elif self.activation == "tanh":
            return 1 - np.tanh(raw_val) ** 2
        return 1.0
    
    def s_layer(self, circuit, x):
        """
        Data encoding layer (S-layer).
        Applies Hadamard gates followed by RY rotations with input features.
        """
        if len(x) != self.n_q:
            raise ValueError(f"Input size {len(x)} does not match n_qubits {self.n_q}")
        
        for i in range(self.n_q):
            circuit.add_gate(H(i))
            circuit.add_gate(RY(i, x[i] * self.multiplier))
    
    def w_layer(self, circuit, p):
        """
        Variational layer (W-layer).
        Applies parameterized RY rotations followed by topology-dependent CNOTs.
        """
        if len(p) != self.n_q:
            raise ValueError(f"Parameter size {len(p)} does not match n_qubits {self.n_q}")
        
        for i in range(self.n_q):
            circuit.add_gate(RY(i, p[i]))
        
        if self.topo == "Circular":
            for i in range(self.n_q - 1): 
                circuit.add_gate(CNOT(i, i + 1))
            circuit.add_gate(CNOT(self.n_q - 1, 0))
        elif self.topo == "Linear":
            for i in range(self.n_q - 1): 
                circuit.add_gate(CNOT(i, i + 1))
        elif self.topo == "Pairwise":
            for i in range(0, self.n_q - 1, 2): 
                circuit.add_gate(CNOT(i, i + 1))
    
    def _get_raw_expectation(self, x, params):
        """
        Compute raw expectation value <ψ|Z⊗Z⊗...⊗Z|ψ> without activation.
        
        Args:
            x (array): Input feature vector
            params (array): Circuit parameters, shape (L+1, n_qubits)
        
        Returns:
            float: Raw expectation value in range [-1, 1]
        """
        state = QuantumState(self.n_q)
        circuit = QuantumCircuit(self.n_q)
        
        # Feature map
        self.s_layer(circuit, x)
        
        # Variational layers with re-uploading
        for i in range(self.L):
            self.w_layer(circuit, params[i])
            self.s_layer(circuit, x)
        
        # Final variational layer
        self.w_layer(circuit, params[-1])
        
        circuit.update_quantum_state(state)
        
        # Observable: Z⊗Z⊗...⊗Z on all qubits, note that this could depend on experiments , you may opt to measure only one qubit as example.
        obs = Observable(self.n_q)
        pauli = " ".join([f"Z {q}" for q in range(self.n_q)])
        obs.add_operator(1.0, pauli)
        
        return obs.get_expectation_value(state)

    def forward(self, x, params):
        """
        Forward pass through the QMLP.
        
        Args:
            x (array): Input feature vector
            params (array): Circuit parameters
        
        Returns:
            float: Model prediction after activation
        """
        raw_val = self._get_raw_expectation(x, params)
        return self._apply_activation(raw_val)
    
    def predict(self, X, params):
        """
        Batch prediction.
        
        Args:
            X (array): Input dataset, shape (N, n_qubits)
            params (array): Circuit parameters
        
        Returns:
            array: Predictions for all samples
        """
        return np.array([self.forward(x, params) for x in X])
    
    def get_gradients(self, x, y, params):
        """
        Compute gradients using parameter shift rule.
        
        Gradients breakdown:
            ∂L/∂θ = (∂L/∂ŷ) × (∂ŷ/∂E) × (∂E/∂θ)
        
        where:
            - ∂L/∂ŷ: loss derivative
            - ∂ŷ/∂E: activation derivative
            - ∂E/∂θ: quantum gradient via parameter shift
        
        Args:
            x (array): Single input sample
            y (float): Target value
            params (array): Circuit parameters, shape (L+1, n_qubits)
        
        Returns:
            tuple: (gradients, loss)
                - gradients: array of shape (L+1, n_qubits)
                - loss: scalar loss value
        """
        # 1. Forward Pass (get raw and activated values)
        raw_val = self._get_raw_expectation(x, params)
        pred = self._apply_activation(raw_val)
        error = pred - y
        
        # 2. Loss Derivative 
        if self.loss == "mse":
            loss = error ** 2
            loss_derivative = 2 * error
        else:  # mae
            loss = np.abs(error)
            loss_derivative = np.sign(error)
        
        # 3. Activation Derivative
        act_derivative = self._activation_derivative(raw_val)
        
        # Pre-multiply classical components of the chain rule
        classical_factor = loss_derivative * act_derivative
        
        grads = np.zeros_like(params)
        
        # If classical grad is zero (e.g. dead ReLU), skip quantum computation
        if abs(classical_factor) < 1e-12:
            return grads, loss

        # 4. Parameter Shift Rule on Raw Expectation Values
        for l in range(params.shape[0]):
            for q in range(params.shape[1]):
                p_plus, p_minus = params.copy(), params.copy()
                p_plus[l, q] += self.shift
                p_minus[l, q] -= self.shift
                
                # Compute shifted expectations
                exp_plus = self._get_raw_expectation(x, p_plus)
                exp_minus = self._get_raw_expectation(x, p_minus)
                
                # Quantum gradient component
                df_quantum = (exp_plus - exp_minus) / 2
                
                # Final Chain Rule
                grads[l, q] = classical_factor * df_quantum
        
        return grads, loss
