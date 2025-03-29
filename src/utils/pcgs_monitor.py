import numpy as np
import logging
import time
import multiprocessing as mp
from queue import Empty
from typing import List, Optional, Dict, Tuple

from utils.process_manager import ProcessManager, Message, MessageType

def prepare_chains_array(chain_samples: Dict[str, List[np.ndarray]], 
                        parameter: str,
                        window_size: Optional[int] = None,
                        thin: int = 1,
                        use_window: bool = False) -> np.ndarray:
    """
    Prepare chain samples into a 3D numpy array (chains × samples × dimensions).
    
    Args:
        chain_samples: Dictionary containing parameter samples from each chain
        parameter: Name of parameter to prepare ('beta' or 'theta')
        window_size: Optional window size for recent samples
        thin: Thinning interval for samples (1 = no thinning)
        use_window: Whether to use window_size to select recent samples (for R-hat)
        
    Returns:
        np.ndarray: 3D array of shape (n_chains, n_samples, n_dimensions)
    """
    # First thin the chains
    thinned_chains = []
    for chain in chain_samples[parameter]:
        thinned = chain[::thin]
        if use_window and window_size is not None:
            thinned = thinned[-window_size:]
        thinned_chains.append(thinned)
    
    # Get thinned chain lengths and find minimum
    chain_lengths = [len(chain) for chain in thinned_chains]
    min_length = min(chain_lengths)
    
    # Apply window size if specified
    if window_size is not None:
        window_size_thinned = window_size // thin  # Adjust window size for thinning
        min_length = min(min_length, window_size_thinned)
    
    # Convert stored lists to numpy arrays
    chains_array = []
    expected_shape = None
    
    for chain_idx, chain in enumerate(thinned_chains):
        # Take only the last min_length samples from this chain
        start_idx = len(chain) - min_length
        chain_arrays = []
        for sample_idx in range(start_idx, len(chain)):
            sample_array = np.array(chain[sample_idx])
            if expected_shape is None:
                expected_shape = sample_array.shape
            elif sample_array.shape != expected_shape:
                raise ValueError(f"Inconsistent shapes in chain {chain_idx}, sample {sample_idx}")
            chain_arrays.append(sample_array)
        chains_array.append(chain_arrays)
    
    # Convert to 3D array and reshape
    chains_array = np.array(chains_array)
    original_shape = chains_array.shape
    flat_shape = (original_shape[0], original_shape[1], -1)
    
    return chains_array.reshape(flat_shape)


def compute_gelman_rubin(chain_samples, parameter, window_size=None):
    """Compute Gelman-Rubin statistic with minimal output."""
    chains_array = prepare_chains_array(chain_samples, parameter, window_size, thin=1)
    
    # Compute statistics
    chain_means = np.mean(chains_array, axis=1)
    B = chains_array.shape[1] * np.var(chain_means, axis=0, ddof=1)
    within_vars = [np.var(chain, axis=0, ddof=1) for chain in chains_array]
    W = np.mean(within_vars, axis=0)
    
    # Calculate potential scale reduction factor with safety checks
    var_hat = ((chains_array.shape[1] - 1) / chains_array.shape[1]) * W + B/chains_array.shape[1]
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        R_hat = np.sqrt(var_hat / (W + epsilon))
        R_hat = np.nan_to_num(R_hat, nan=np.inf)
    
    return np.max(R_hat)
   


def compute_effective_sample_size(samples: Dict[str, List[np.ndarray]], 
                                parameter: str,
                                window_size: Optional[int] = None) -> float:
    """Compute effective sample size using autocorrelation method."""
    chains_array = prepare_chains_array(samples, parameter, window_size, thin=1)
    n_chains, n_samples, n_dims = chains_array.shape
    
    # Initialize ESS array for each dimension
    ess_per_dim = np.zeros(n_dims)
    
    for dim in range(n_dims):
        # Extract data for current dimension across all chains
        dim_data = chains_array[:, :, dim]
        
        # Center the data
        centered = dim_data - dim_data.mean(axis=1, keepdims=True)
        
        # Compute variance
        variance = np.mean(np.var(centered, axis=1))
        
        if variance == 0:
            ess_per_dim[dim] = n_chains * n_samples
            continue
            
        # Compute autocorrelations
        max_lag = min(n_samples // 3, 100)
        rho = np.zeros(max_lag)
        
        for lag in range(max_lag):
            cov = np.mean([
                np.mean(centered[:, lag:] * centered[:, :n_samples-lag])
                for chain in range(n_chains)
            ])
            rho[lag] = cov / variance
        
        # Find where autocorrelation drops below 0.05 or becomes negative
        cutoff = np.where((np.abs(rho) < 0.05) | (rho < 0))[0]
        max_lag = cutoff[0] if len(cutoff) > 0 else max_lag
        
        # Compute ESS
        tau = -1 + 2 * np.sum(rho[:max_lag])
        ess_per_dim[dim] = n_chains * n_samples / (1 + tau)
    
    return np.median(ess_per_dim)


def monitor_convergence(monitor_queues: List[mp.Queue],
                       control_queues: List[mp.Queue],
                       monitor_queue: mp.Queue,
                       process_manager: ProcessManager,
                       r_hat_threshold: float = 1.1,
                       max_iterations: Optional[int] = None,
                       window_size: Optional[int] = None,
                       calculate_ess: bool = False,
                       ess_threshold: float = 400):
    """Monitor chain convergence and coordinate sample collection."""
    
    monitor = ChainMonitor(
        num_chains=len(monitor_queues),
        r_hat_threshold=r_hat_threshold,
        max_iterations=max_iterations,
        window_size=window_size,
        calculate_ess=calculate_ess,
        ess_threshold=ess_threshold
    )
    
    try:
        while True:
            # Health check
            if not process_manager.check_heartbeats():
                raise RuntimeError("One or more chains have failed")
            
            # Process incoming messages
            new_samples = monitor.process_messages(monitor_queues, process_manager)
            
            # Check completion
            if monitor.is_collection_complete():
                break
                
            # Check convergence if we have new samples
            if new_samples and not monitor.is_collecting():
                monitor.check_convergence()
                
                # Start collection if criteria met
                if monitor.should_start_collection():
                    monitor.start_collection(control_queues)
            
            time.sleep(0.1)
            
        # Send final statistics
        monitor_queue.put(monitor.get_statistics())
        
    except Exception as e:
        logging.error(f"Monitor error: {str(e)}")
        raise

class ChainState:
    def __init__(self, store_all_samples: bool = False):
        self.current_iteration = 0
        self.samples = {'beta': [], 'theta': []} if store_all_samples else None
        self.recent_samples = {'beta': [], 'theta': []}  # For R-hat calculation
        self.collecting = False
        self.completed = False
        self.store_all_samples = store_all_samples
        
    def update_iteration(self, iteration: int):
        self.current_iteration = max(self.current_iteration, iteration)
        
    def add_sample(self, data: Dict, window_size: Optional[int] = None):
        for param in ['beta', 'theta']:
            if param in data:
                if self.store_all_samples:
                    self.samples[param].append(data[param])
                    
                # Always maintain recent samples for R-hat
                self.recent_samples[param].append(data[param])
                if window_size and len(self.recent_samples[param]) > window_size:
                    self.recent_samples[param].pop(0)

class ChainMonitor:
    def __init__(self, num_chains: int, r_hat_threshold: float,
                 max_iterations: Optional[int] = None,
                 window_size: Optional[int] = None,
                 calculate_ess: bool = False,
                 ess_threshold: float = 400):
        """Initialize chain monitoring state."""
        self.num_chains = num_chains
        self.r_hat_threshold = r_hat_threshold
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.calculate_ess = calculate_ess
        self.ess_threshold = ess_threshold
        
        # Per-chain state tracking
        self.chain_states = {i: ChainState(store_all_samples=calculate_ess) 
                           for i in range(num_chains)}
        self.r_hats = {}
        self.ess_values = {} if calculate_ess else None
        self.r_hat_checks = 0
        
    def process_messages(self, monitor_queues: List[mp.Queue], 
                        process_manager: ProcessManager) -> bool:
        """Process incoming messages from all chains. Returns True if new samples received."""
        new_samples = False
        
        for chain_id, queue in enumerate(monitor_queues):
            try:
                while not queue.empty():
                    msg = queue.get_nowait()
                    
                    if msg.type == MessageType.ERROR:
                        raise RuntimeError(f"Chain {chain_id} error: {msg.error}")
                    elif msg.type == MessageType.HEARTBEAT:
                        if msg.iteration is not None:
                            self.chain_states[chain_id].update_iteration(msg.iteration)
                        process_manager.update_heartbeat(msg.chain_id)
                        
                    elif msg.type == MessageType.SAMPLE and not self.chain_states[chain_id].collecting:
                        if msg.iteration is not None:
                            self.chain_states[chain_id].update_iteration(msg.iteration)
                        self.chain_states[chain_id].add_sample(msg.data, self.window_size)
                        new_samples = True
                        
                    elif msg.type == MessageType.COLLECTION_COMPLETE:
                        print(f"Chain {chain_id} completed collection")
                        self.chain_states[chain_id].completed = True
                        
            except Empty:
                continue
            except Exception as e:
                process_manager.handle_chain_failure(chain_id, str(e))
                raise
                
        return new_samples
    
    def check_convergence(self) -> Tuple[float, float]:
        """Compute R-hat and optionally ESS values."""
        if not all(len(state.recent_samples['beta']) >= 2 and 
                  len(state.recent_samples['theta']) >= 2 
                  for state in self.chain_states.values()):
            return float('inf'), 0.0
        
        # Prepare samples for R-hat calculation
        recent_samples = {
            'beta': [state.recent_samples['beta'] for state in self.chain_states.values()],
            'theta': [state.recent_samples['theta'] for state in self.chain_states.values()]
        }
        
        # Compute R-hat for each parameter
        for param in recent_samples:
            self.r_hats[param] = compute_gelman_rubin(recent_samples, param, self.window_size)
        
        # Compute ESS if enabled
        if self.calculate_ess:
            full_samples = {
                'beta': [state.samples['beta'] for state in self.chain_states.values()],
                'theta': [state.samples['theta'] for state in self.chain_states.values()]
            }
            for param in full_samples:
                self.ess_values[param] = compute_effective_sample_size(full_samples, param)
        
        max_r_hat = max(self.r_hats.values()) if self.r_hats else float('inf')
        min_ess = min(self.ess_values.values()) if self.ess_values else float('inf')
        
        print(f"Check #{self.r_hat_checks}: "
              f"max R̂ = {max_r_hat:.4f} (threshold: {self.r_hat_threshold:.4f})"
              + (f", min ESS = {min_ess:.1f} (threshold: {self.ess_threshold:.1f})" 
                 if self.calculate_ess else ""))
        
        self.r_hat_checks += 1
        
        return max_r_hat, min_ess
    
    def should_start_collection(self) -> bool:
        """Determine if collection phase should start."""
        if any(state.collecting for state in self.chain_states.values()):
            return False
            
        # Check if all chains have reached max iterations
        if self.max_iterations:
            iterations = [state.current_iteration for state in self.chain_states.values()]
            print(f"Current chain iterations: {iterations}")
            if all(iter_num >= self.max_iterations for iter_num in iterations):
                print(f"All chains reached max iterations ({self.max_iterations})")
                return True
        
        # Check R-hat convergence
        if not self.r_hats:
            return False
        
        r_hat_converged = max(self.r_hats.values()) < self.r_hat_threshold
        
        # If ESS calculation is enabled, check ESS convergence too
        if self.calculate_ess:
            if not self.ess_values:
                return False
            ess_sufficient = min(self.ess_values.values()) > self.ess_threshold
            return r_hat_converged and ess_sufficient
        
        return r_hat_converged
    
    def start_collection(self, control_queues: List[mp.Queue]):
        """Start the collection phase for all chains."""
        if any(state.collecting for state in self.chain_states.values()):
            return
            
        for q in control_queues:
            q.put(Message(
                type=MessageType.START_COLLECTION,
                chain_id=-1
            ))
        
        for state in self.chain_states.values():
            state.collecting = True
    
    def is_collecting(self) -> bool:
        """Check if any chain is in collection phase."""
        return any(state.collecting for state in self.chain_states.values())
    
    def is_collection_complete(self) -> bool:
        """Check if all chains have completed collection."""
        return all(state.completed for state in self.chain_states.values())
    
    def get_statistics(self) -> Dict:
        """Get final monitoring statistics."""
        return {
            'r_hats': self.r_hats,
            'ess_values': self.ess_values,
            'r_hat_checks': self.r_hat_checks,
            'chain_iterations': {i: state.current_iteration 
                               for i, state in self.chain_states.items()},
            'max_iterations_reached': (self.max_iterations and 
                any(state.current_iteration >= self.max_iterations 
                    for state in self.chain_states.values()))
        }

