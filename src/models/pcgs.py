import numpy as np
from numpy.random import choice
import multiprocessing as mp
import time
import queue
import logging

from utils.process_manager import ProcessManager, Message, MessageType
from typing import Optional, List, Dict
from utils.pcgs_monitor import monitor_convergence



def initialize(W, num_topics):
    # Initialize topic assignments z randomly
    z = np.random.choice(num_topics, size=W.shape)

    # Initialize document-topic counts
    n = np.zeros((W.shape[0], num_topics), dtype=int)
    for s in range(W.shape[0]):
        n[s] = np.bincount(z[s], minlength=num_topics)

    return z, n


def update_beta(W, z, num_topics):
    counts = np.zeros((num_topics, W.shape[1]))

    for k in range(num_topics):
        # Filter W where the topic assignment z equals k
        indices = np.where(z == k)[0]
        filtered_W = W[indices, :]  # Ensure this slices as a matrix
        counts[k] = filtered_W.sum(axis=0)

    total_counts = counts.sum(axis=1)
    subject_counts = np.array([len(np.where(z == k)[0]) for k in range(num_topics)])


    # Handle division by zero and update beta
    beta = np.zeros_like(counts)
    for k in range(num_topics):
        if total_counts[k] == 0:
            beta[k] = np.ones(W.shape[1]) * 0.1  # Handling division by zero with a default small number
        else:
            beta[k] = counts[k] / subject_counts[k]

    return beta

def update_theta(n, alpha):
   
   # Sum over topics to get total counts per subject
    total_counts_per_subject = n.sum(axis=1) + alpha.sum()
    
    # Add alpha to counts and normalize by total counts per subject
    theta = (n + alpha) / total_counts_per_subject[:, None]  # Using broadcasting
    
    return theta


def compute_conditional(s, j, W, n, alpha, beta):
    w_sj = W[s, j]  # Diagnosis indicator for subject s and disease j

    # Vectorized computation for each topic
    p = (beta[:, j]**w_sj * (1 - beta[:, j])**(1 - w_sj)) * (alpha + n[s, :])

    # Normalize to get a valid probability distribution
    p /= p.sum()
    return p



class GibbsSampler:
    def __init__(self, W: np.ndarray, alpha: np.ndarray, num_topics: int):
        self.W = W
        self.alpha = alpha
        self.num_topics = num_topics
        self.z, self.n = initialize(W, num_topics)
        self.beta = update_beta(W, self.z, num_topics)
        
    def run_iteration(self) -> dict:
        """Run a single Gibbs sampling iteration."""
        for s in range(self.W.shape[0]):
            self._update_topic_assignments(s)
        return self._get_current_state()
    
    def _update_topic_assignments(self, subject: int):
        """Update topic assignments for a single subject."""
        self.beta = update_beta(self.W, self.z, self.num_topics)
        p = np.array([compute_conditional(subject, j, self.W, self.n, self.alpha, self.beta) 
                     for j in range(self.W.shape[1])])
        
        for j in range(self.W.shape[1]):
            k_old = self.z[subject, j]
            self.n[subject, k_old] -= 1
            k_new = choice(np.arange(self.num_topics), p=p[j])
            self.z[subject, j] = k_new
            self.n[subject, k_new] += 1
    
    def _get_current_state(self) -> dict:
        """Get current parameter estimates."""
        return {
            'beta': self.beta.copy(),
            'theta': update_theta(self.n, self.alpha),
            'z': self.z.copy()
        }

class ChainRunner:
    def __init__(self, chain_id: int, sampler: GibbsSampler, 
                 monitor_queue: mp.Queue, final_result_queue: mp.Queue, 
                 control_queue: mp.Queue, max_iterations: int,
                 sample_freq: int = 10, post_convergence_samples: int = 50):
        self.chain_id = chain_id
        self.sampler = sampler
        self.monitor_queue = monitor_queue
        self.final_result_queue = final_result_queue
        self.control_queue = control_queue
        self.max_iterations = max_iterations
        self.sample_freq = sample_freq
        self.post_convergence_samples = post_convergence_samples
        
        self.collecting_samples = False
        self.samples_collected = 0
        self.final_samples = {
            'beta_samples': [],
            'theta_samples': [],
            'z_samples': []
        }
    
    def run(self):
        """Main chain execution loop."""
        try:
            self._send_initial_heartbeat()
            
            for iter_num in range(self.max_iterations + self.post_convergence_samples):
                current_state = self.sampler.run_iteration()
                self._handle_messaging(iter_num, current_state)
                
                if self._should_stop():
                    break
            
            self._complete_collection()
            self._send_final_results()
            
        except Exception as e:
            self._handle_error(e)
            raise
    
    # Operation methods
    def _handle_messaging(self, iter_num: int, current_state: dict):
        """Handle messaging and sample collection."""
        if iter_num % self.sample_freq == 0:
            self._send_heartbeat(iter_num)
            if not self.collecting_samples:
                self._handle_monitoring(iter_num, current_state)
        
        # Check control messages
        self._check_control_messages()
    
    def _complete_collection(self):
        """Ensure collection completes even if max_iterations reached."""
        while self.collecting_samples and self.samples_collected < self.post_convergence_samples:
            current_state = self.sampler.run_iteration()
            self._collect_samples(current_state)
            
        if self.samples_collected > 0:
            self.monitor_queue.put(Message(
                type=MessageType.COLLECTION_COMPLETE,
                chain_id=self.chain_id
            ))
            
    def _send_final_results(self):
        """Send final results if samples were collected."""
        if self.samples_collected > 0:
            print(f"Chain {self.chain_id} sending final results")
            self.final_result_queue.put(Message(
                type=MessageType.COLLECTION_COMPLETE,
                chain_id=self.chain_id,
                data=self.final_samples
            ))
    
    def _send_heartbeat(self, iter_num: int):
        if iter_num % self.sample_freq == 0:
            self.monitor_queue.put(Message(
                type=MessageType.HEARTBEAT,
                chain_id=self.chain_id,
                iteration=iter_num
            ))
    
    def _check_control_messages(self):
        try:
            while not self.control_queue.empty():
                msg = self.control_queue.get_nowait()
                if isinstance(msg, Message) and msg.type == MessageType.START_COLLECTION:
                    self.collecting_samples = True
                    print(f"Chain {self.chain_id} starting collection phase")
        except queue.Empty:
            pass
    
    def _handle_monitoring(self, iter_num: int, state: dict):
        """Send monitoring data to monitor queue."""
        if not self.collecting_samples:
            self.monitor_queue.put(Message(
                type=MessageType.SAMPLE,
                chain_id=self.chain_id,
                data={'beta': state['beta'], 'theta': state['theta']},
                iteration=iter_num
            ))
    
    def _collect_samples(self, state: dict):
        if self.collecting_samples:
            print(f"Chain {self.chain_id} collecting sample "
                  f"{self.samples_collected + 1}/{self.post_convergence_samples}")
            for key in self.final_samples:
                self.final_samples[key].append(state[key.split('_')[0]])
            self.samples_collected += 1
    
    # Helper methods
    def _send_initial_heartbeat(self):
        """Send initial heartbeat message."""
        self.monitor_queue.put(Message(
            type=MessageType.HEARTBEAT,
            chain_id=self.chain_id,
            iteration=0
        ))
        
    def _should_stop(self) -> bool:
        return self.collecting_samples and self.samples_collected >= self.post_convergence_samples
    
    def _handle_error(self, error: Exception):
        """Handle chain errors by sending error message."""
        self.monitor_queue.put(Message(
            type=MessageType.ERROR,
            chain_id=self.chain_id,
            error=str(error)
        ))


def run_chain(chain_id: int, W: np.ndarray, alpha: np.ndarray, 
              num_topics: int, max_iterations: int, 
              monitor_queue: mp.Queue,
              final_result_queue: mp.Queue,
              control_queue: mp.Queue, 
              sample_freq: int = 10,
              post_convergence_samples: int = 50):
    """Run a single Gibbs sampling chain."""
    sampler = GibbsSampler(W, alpha, num_topics)
    runner = ChainRunner(
        chain_id=chain_id,
        sampler=sampler,
        monitor_queue=monitor_queue,
        final_result_queue=final_result_queue,
        control_queue=control_queue,
        max_iterations=max_iterations,
        sample_freq=sample_freq,
        post_convergence_samples=post_convergence_samples
    )
    runner.run()

class GibbsSamplingCoordinator:
    def __init__(self, W: np.ndarray, alpha: np.ndarray, num_topics: int,
                 num_chains: int, max_iterations: Optional[int] = None,
                 window_size: Optional[int] = None, r_hat_threshold: float = 1.1,
                 ess_threshold: float = 400, post_convergence_samples: int = 50):
        """Initialize sampling coordinator with configuration parameters."""
        self.W = W
        self.alpha = alpha
        self.num_topics = num_topics
        self.num_chains = num_chains
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.r_hat_threshold = r_hat_threshold
        self.ess_threshold = ess_threshold
        self.post_convergence_samples = post_convergence_samples
        
        self._validate_params()
        
        # Process management
        self.process_manager = ProcessManager(num_chains=num_chains)
        # Use process manager's queues
        self.monitor_queues = self.process_manager.monitor_queues
        self.control_queues = self.process_manager.control_queues
        self.result_queues = self.process_manager.result_queues
        self.monitor_queue = self.process_manager.monitor_queue
        
    def run(self) -> Dict:
        """Execute the parallel Gibbs sampling process."""
        try:
            self._setup_processes()
            results = self._collect_results()
            return results
        finally:
            self._cleanup()
            
    def _validate_params(self):
        """Validate input parameters."""
        if self.W.ndim != 2:
            raise ValueError("W must be a 2D array")
        if self.alpha.shape[0] != self.num_topics:
            raise ValueError("Alpha length must match number of topics")
        if self.num_chains < 2:
            raise ValueError("At least 2 chains required for convergence monitoring")
            
    def _setup_processes(self):
        """Set up and start all processes."""
        self._start_monitor()
        self._start_chains()
        
    def _start_monitor(self):
        """Initialize and start the monitor process."""
        self.monitor_process = mp.Process(
            target=monitor_convergence,
            args=(
                self.monitor_queues,
                self.control_queues,
                self.monitor_queue,
                self.process_manager,
                self.r_hat_threshold,
                self.max_iterations,
                self.window_size,
                self.ess_threshold
            )
        )
        self.monitor_process.start()
        
    def _start_chains(self):
        """Initialize and start all chain processes."""
        self.chain_processes = []
        for i in range(self.num_chains):
            p = mp.Process(
                target=run_chain,
                args=(
                    i, self.W, self.alpha, self.num_topics,
                    self.max_iterations, self.monitor_queues[i],
                    self.result_queues[i], self.control_queues[i]
                ),
                kwargs={'post_convergence_samples': self.post_convergence_samples}
            )
            p.start()
            self.chain_processes.append(p)
            
    def _collect_results(self) -> Dict:
        """Collect and aggregate results from all chains."""
        try:
            # Wait for monitor to complete
            monitor_stats = self.monitor_queue.get()
            
            # Collect chain results
            chain_results = []
            for chain_id in range(self.num_chains):
                result = self.result_queues[chain_id].get()
                if isinstance(result, Message):
                    if result.type == MessageType.ERROR:
                        raise RuntimeError(f"Chain error: {result.error}")
                    elif result.type == MessageType.COLLECTION_COMPLETE:
                        chain_results.append(result.data)  # Append to list instead of dict
                    else:
                        raise RuntimeError(f"Unexpected message type from chain {chain_id}: {result.type}")
            
            return chain_results  # Return just the list of chain results
            
        except Exception as e:
            logging.error(f"Error collecting results: {str(e)}")
            raise
            
    def _cleanup(self):
        """Clean up processes and queues."""
        # Terminate all processes
        for p in getattr(self, 'chain_processes', []):
            if p.is_alive():
                p.terminate()
        if hasattr(self, 'monitor_process') and self.monitor_process.is_alive():
            self.monitor_process.terminate()
            
        # Close all queues
        for q in self.monitor_queues + self.control_queues + self.result_queues:
            q.close()
        self.monitor_queue.close()

def collapsed_gibbs_sampling(W: np.ndarray, alpha: np.ndarray, num_topics: int,
                           num_chains: int, max_iterations: Optional[int] = None,
                           window_size: Optional[int] = None, r_hat_threshold: float = 1.1,
                           ess_threshold: float = 400, post_convergence_samples: int = 50) -> Dict:
    """Execute parallel collapsed Gibbs sampling with convergence monitoring."""
    coordinator = GibbsSamplingCoordinator(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        num_chains=num_chains,
        max_iterations=max_iterations,
        window_size=window_size,
        r_hat_threshold=r_hat_threshold,
        ess_threshold=ess_threshold,
        post_convergence_samples=post_convergence_samples
    )
    return coordinator.run()
