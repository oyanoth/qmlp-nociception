import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, RY, CNOT
from tqdm import tqdm


class EntanglementAnalyzer:
    """
    Analyzes maximum bipartite entanglement entropy of quantum circuits using Qulacs on trained parameters.
    """
    
    def __init__(self, n_qubits, topology="Circular", layers=2, 
                 use_scaler=False, scaling_angle=np.pi):
        """
        Initialize entanglement analyzer.
        
        Args:
            n_qubits (int): Number of qubits
            topology (str): "Circular", "Linear", or "Pairwise"
            layers (int): Number of variational layers
            use_scaler (bool): Whether to scale data encoding
            scaling_angle (float): Scaling factor for angle embedding
        """
        self.n_qubits = n_qubits
        self.topology = topology
        self.layers = layers
        self.multiplier = scaling_angle if use_scaler else 1.0
        
        # Compute Haar (Page) entropy for normalization
        n_A = n_qubits // 2
        n_B = n_qubits - n_A
        self.S_haar_max = (n_A * np.log(2) - 0.5) if n_A <= n_B else (n_B * np.log(2) - 0.5)
    
    def _s_layer(self, circuit, x):
        """Data encoding layer: Hadamard + AngleEmbedding with RY."""
        for i in range(self.n_qubits):
            circuit.add_gate(H(i))
            circuit.add_gate(RY(i, x[i] * self.multiplier))
    
    def _w_layer(self, circuit, params):
        """Variational layer with topology-dependent entanglement."""
        for i in range(self.n_qubits):
            circuit.add_gate(RY(i, params[i]))
        
        if self.topology == "Circular":
            for i in range(self.n_qubits - 1):
                circuit.add_gate(CNOT(i, i + 1))
            circuit.add_gate(CNOT(self.n_qubits - 1, 0))
        
        elif self.topology == "Linear":
            for i in range(self.n_qubits - 1):
                circuit.add_gate(CNOT(i, i + 1))
        
        elif self.topology == "Pairwise":
            for i in range(0, self.n_qubits - 1, 2):
                circuit.add_gate(CNOT(i, i + 1))
    
    def _build_circuit(self, x, params):
        """Build full quantum circuit matching QMLP architecture."""
        circuit = QuantumCircuit(self.n_qubits)
        
        # Initial encoding
        self._s_layer(circuit, x)
        
        # Variational layers with re-encoding
        for layer_idx in range(self.layers):
            self._w_layer(circuit, params[layer_idx])
            self._s_layer(circuit, x)
        
        # Final variational layer
        self._w_layer(circuit, params[-1])
        
        return circuit
    
    def _compute_reduced_density_matrix(self, state_vector, cut):
        """
        Compute reduced density matrix for qubits [0:cut].
        
        Args:
            state_vector (array): Full quantum state
            cut (int): Bipartition point
        
        Returns:
            array: Reduced density matrix ρ_A
        """
        # Reshape to tensor form
        state_tensor = np.reshape(state_vector, [2] * self.n_qubits)
        state_conj = np.conj(state_tensor)
        
        # Partial trace over subsystem B = [cut:n_qubits]
        axes_to_trace = list(range(cut, self.n_qubits))
        rho_tensor = np.tensordot(state_tensor, state_conj, 
                                   axes=[axes_to_trace, axes_to_trace])
        
        # Reshape to matrix
        dim_A = 2 ** cut
        rho_A = np.reshape(rho_tensor, (dim_A, dim_A))
        
        return rho_A
    
    def _von_neumann_entropy(self, rho):
        """
        Compute von Neumann entropy S = -Tr(ρ log ρ).
        
        Args:
            rho (array): Density matrix
        
        Returns:
            float: Von Neumann entropy
        """
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy
    
    def analyze_sample(self, x, params):
        """
        Analyze entanglement for a single input sample.
        
        Args:
            x (array): Input feature vector (length = n_qubits)
            params (array): Circuit parameters (shape = [layers+1, n_qubits])
        
        Returns:
            dict: Max entropy, normalized value, and bond index
        """
        # Create state and apply circuit
        state = QuantumState(self.n_qubits)
        circuit = self._build_circuit(x, params)
        circuit.update_quantum_state(state)
        
        # Get state vector
        state_vector = state.get_vector()
        
        # Compute entropy for all bipartitions
        entropies = []
        for cut in range(1, self.n_qubits):
            rho_A = self._compute_reduced_density_matrix(state_vector, cut)
            S = self._von_neumann_entropy(rho_A)
            entropies.append(S)
        
        entropies = np.array(entropies)
        
        # Find maximum
        max_entropy = np.max(entropies)
        max_bond = np.argmax(entropies) + 1
        normalized_max = max_entropy / self.S_haar_max
        
        return {
            'max_entropy': max_entropy,
            'normalized_max': normalized_max,
            'max_bond': max_bond,
            'S_haar_max': self.S_haar_max
        }
    
    def analyze_dataset(self, X, params, n_samples=1000, verbose=True):
        """
        Analyze entanglement averaged over multiple samples.
        
        Args:
            X (array): Input dataset (shape = [n_samples, n_qubits])
            params (array): Trained circuit parameters
            n_samples (int): Number of samples to analyze
            verbose (bool): Show progress bar
        
        Returns:
            dict: Statistics over the dataset
        """
        n_samples = min(n_samples, len(X))
        max_entropies = []
        normalized_maxs = []
        max_bonds = []
        
        iterator = tqdm(range(n_samples), desc=f"Analyzing {self.topology}", disable=not verbose)
        
        for idx in iterator:
            result = self.analyze_sample(X[idx], params)
            max_entropies.append(result['max_entropy'])
            normalized_maxs.append(result['normalized_max'])
            max_bonds.append(result['max_bond'])
        
        max_entropies = np.array(max_entropies)
        normalized_maxs = np.array(normalized_maxs)
        
        results = {
            'mean_max_entropy': np.mean(max_entropies),
            'std_max_entropy': np.std(max_entropies),
            'mean_normalized_max': np.mean(normalized_maxs),
            'std_normalized_max': np.std(normalized_maxs),
            'S_haar_max': self.S_haar_max,
            'topology': self.topology,
            'n_samples': n_samples,
            'n_qubits': self.n_qubits,
            'layers': self.layers
        }
        
        return results


def compare_topologies(X, params_dict, n_qubits, layers=2, 
                       use_scaler=False, scaling_angle=np.pi, n_samples=1000, verbose=True):
    """
    Compare entanglement across different circuit topologies.
    
    Args:
        X (array): Input dataset
        params_dict (dict): Dictionary mapping topology names to parameters
            Example: {"Circular": params1, "Linear": params2}
        n_qubits (int): Number of qubits
        layers (int): Number of layers
        use_scaler (bool): Whether scaling is used
        scaling_angle (float): Angle scaling factor
        n_samples (int): Number of samples to analyze
        verbose (bool): Show progress bars
    
    Returns:
        dict: Comparison results for all topologies
    """
    comparison = {}
    
    for topo_name, params in params_dict.items():
        analyzer = EntanglementAnalyzer(
            n_qubits=n_qubits,
            topology=topo_name,
            layers=layers,
            use_scaler=use_scaler,
            scaling_angle=scaling_angle
        )
        
        results = analyzer.analyze_dataset(X, params, n_samples=n_samples, verbose=verbose)
        comparison[topo_name] = results
    
    return comparison