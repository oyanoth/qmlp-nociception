import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil
import os
import torch

def _process_batch(model, X_batch, y_batch, params):
    """
    Compute gradients and loss for a subset of samples.

    Args:
        model: QMLP model instance. Must implement:
               - get_gradients(x, y, params) -> (grad, loss)
        X_batch (ndarray): Subset of input samples, shape (B, n_qubits)
        y_batch (ndarray): Corresponding targets, shape (B,)
        params (ndarray): Current model parameters

    Returns:
        mean_grad (ndarray): Average gradient over the batch
        mean_loss (float): Average loss over the batch
    """
    grads_list = []
    losses = []

    for i in range(len(X_batch)):
        grad, loss = model.get_gradients(X_batch[i], y_batch[i], params)
        grads_list.append(grad)
        losses.append(loss)

    return np.mean(grads_list, axis=0), np.mean(losses)

def _process_forward_batch(model, X_batch, params):
    """
    Forward pass only (no gradients), used for validation.
    
    Args:
        model: QMLP model instance
        X_batch (ndarray): Batch of inputs
        params (ndarray): Model parameters
    
    Returns:
        predictions (list): Model outputs for each sample
    """
    predictions = []
    for i in range(len(X_batch)):
        pred = model.forward(X_batch[i], params)
        predictions.append(pred)
    return predictions

class AdamTrainer:
    """
    Trainer for QMLP using Adam optimizer with mini-batch gradient descent.
    
    Features:
        - Mini-batch training with configurable batch size
        - Parallel gradient computation using joblib
        - Early stopping with validation monitoring
        - Memory usage tracking
        - PyTorch Adam optimizer for momentum and adaptive learning rates
    
    Args:
        model: QMLP model instance
        lr (float): Learning rate for Adam
        betas (tuple): Adam momentum coefficients (beta1, beta2)
        eps (float): Numerical stability constant
    """
    def __init__(self, model, lr=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.process = psutil.Process(os.getpid())
        
    def _get_total_memory(self):
        """Get memory usage including all child processes."""
        try:
            mem = self.process.memory_info().rss / (1024**2)
            for child in self.process.children(recursive=True):
                try:
                    mem += child.memory_info().rss / (1024**2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return mem
        except Exception:
            return 0.0
        
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        n_jobs=-1,
        patience=None,
        validation_freq=1,
    ):
        """
        Train the QMLP model using mini-batch Adam optimization.
        
        Args:
            X_train (ndarray): Training inputs, shape (N, n_qubits)
            y_train (ndarray): Training targets, shape (N,)
            X_val (ndarray or None): Validation inputs
            y_val (ndarray or None): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Number of samples per batch
            n_jobs (int): Number of parallel workers (-1 = all CPUs)
            patience (int or None): Early stopping patience (epochs)
            validation_freq (int): Validate every N epochs
        
        Returns:
            best_params (ndarray): Parameters with best validation loss
            peak_mem (float): Peak RAM usage in MB
        """
        # Validate inputs
        if X_train.shape[1] != self.model.n_q:
            raise ValueError(f"X_train features ({X_train.shape[1]}) != model n_qubits ({self.model.n_q})")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train samples ({len(X_train)}) != y_train samples ({len(y_train)})")
        
        n_samples, n_q = X_train.shape
        steps = int(np.ceil(n_samples / batch_size))
        total_steps = epochs * steps
        
        # Initialize parameters
        params = np.random.uniform(0, 2 * np.pi, size=(self.model.L + 1, n_q))
        
        # Create PyTorch tensor and optimizer
        params_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=False)
        optimizer = torch.optim.Adam([params_tensor], lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps)
        
        # Memory tracking
        baseline_mem = self._get_total_memory()
        peak_mem = baseline_mem
        
        # Early stopping
        best_loss = np.inf
        best_params = params.copy()
        wait = 0
        
        # Progress bar
        pbar = tqdm(
            total=total_steps,
            desc="Training QMLP",
            unit="batch",
            leave=True,
            ncols=0 
        )
        
        # Determine number of workers
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        
        with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
            for epoch in range(1, epochs + 1):
                # Shuffle data
                perm = np.random.permutation(n_samples)
                epoch_loss = 0.0
                
                for s in range(steps):
                    # Get mini-batch
                    idx = perm[s * batch_size : (s + 1) * batch_size]
                    X_batch = X_train[idx]
                    y_batch = y_train[idx]
                    
                    # Get current params as numpy array
                    params = params_tensor.detach().numpy()
                    
                    # Parallelize across workers
                    n_workers = min(n_jobs, len(idx))
                    if n_workers == 0:
                        n_workers = 1
                    
                    chunk_size = len(idx) // n_workers
                    
                    # Compute gradients in parallel
                    res = parallel(
                        delayed(_process_batch)(
                            self.model,
                            X_batch[i * chunk_size : (i + 1) * chunk_size],
                            y_batch[i * chunk_size : (i + 1) * chunk_size],
                            params
                        )
                        for i in range(n_workers)
                    )
                    
                    # Handle remainder samples
                    remainder_start = n_workers * chunk_size
                    if remainder_start < len(idx):
                        rem_grad, rem_loss = _process_batch(
                            self.model,
                            X_batch[remainder_start:],
                            y_batch[remainder_start:],
                            params
                        )
                        grads = np.mean([r[0] for r in res] + [rem_grad], axis=0)
                        batch_loss = np.mean([r[1] for r in res] + [rem_loss])
                    else:
                        grads = np.mean([r[0] for r in res], axis=0)
                        batch_loss = np.mean([r[1] for r in res])
                    
                    # Update parameters using PyTorch Adam
                    optimizer.zero_grad()
                    params_tensor.grad = torch.tensor(grads, dtype=torch.float32)
                    optimizer.step()
                    
                    # Accumulate loss
                    epoch_loss += batch_loss * len(idx)
                    
                    # Update memory tracking
                    current_mem = self._get_total_memory()
                    peak_mem = max(peak_mem, current_mem)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"ep={epoch}/{epochs} | bl={batch_loss:.4f} | best={best_loss:.4f} | RAMp={peak_mem:.0f}MB",
                        refresh=False
                    )
                
                # ----------------- END OF EPOCH -----------------
                curr_train_loss = epoch_loss / n_samples
                
                # Validation
                if (
                    X_val is not None
                    and y_val is not None
                    and epoch % validation_freq == 0
                ):
                    # Get current params for validation
                    params = params_tensor.detach().numpy()
                    
                    # Parallelize validation - ensure we process ALL samples
                    n_workers = min(n_jobs, len(X_val))
                    if n_workers == 0:
                        n_workers = 1
                    
                    chunk_size = len(X_val) // n_workers
                    
                    # Build chunks ensuring all samples are covered
                    chunks = []
                    for i in range(n_workers):
                        start_idx = i * chunk_size
                        # Last worker gets all remaining samples
                        if i == n_workers - 1:
                            end_idx = len(X_val)
                        else:
                            end_idx = (i + 1) * chunk_size
                        
                        if start_idx < len(X_val):
                            chunks.append((start_idx, end_idx))
                    
                    res = parallel(
                        delayed(_process_forward_batch)(
                            self.model,
                            X_val[start:end],
                            params
                        )
                        for start, end in chunks
                    )
                    
                    # Flatten predictions
                    y_val_pred = []
                    for r in res:
                        y_val_pred.extend(r)
                    
                    # Verify we got all predictions
                    if len(y_val_pred) != len(y_val):
                        raise RuntimeError(
                            f"Validation prediction count mismatch: "
                            f"got {len(y_val_pred)} predictions for {len(y_val)} samples"
                        )
                    
                    # Update memory tracking
                    current_mem = self._get_total_memory()
                    peak_mem = max(peak_mem, current_mem)
                    
                    # Calculate validation loss
                    y_val_pred = np.array(y_val_pred)
                    if self.model.loss == "mse":
                        monitor_loss = np.mean((y_val_pred - y_val) ** 2)
                    else:  # mae
                        monitor_loss = np.mean(np.abs(y_val_pred - y_val))
                    
                    # Early stopping check
                    if monitor_loss < best_loss:
                        best_loss = monitor_loss
                        best_params = params_tensor.detach().numpy().copy()
                        wait = 0
                    else:
                        wait += 1
                    
                    if patience is not None and wait >= patience:
                        pbar.write(f"\nEarly stopping triggered at epoch {epoch}")
                        break
        
        pbar.close()
        return best_params, peak_mem