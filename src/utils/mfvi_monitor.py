import numpy as np

class ELBOMonitor:
    def __init__(self, convergence_threshold: float = 1e-4, max_iterations: int = 1000):
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.elbo_history = []
        self.iteration = 0
        
        # For debugging/analysis
        self.parameter_changes = {
            'e_theta': [],
            'e_log_theta': [],
            'z': [],
            'beta': []
        }
    
    def check_convergence(self, elbo_before: float, elbo_after: float, param_changes: dict) -> bool:
        """Check if ELBO has converged."""
        self.elbo_history.append(elbo_after)
        self.iteration += 1
        
        # Record parameter changes
        for param, change in param_changes.items():
            self.parameter_changes[param].append(change)
        
        elbo_change = abs(elbo_after - elbo_before)
        print(f"Iteration {self.iteration}: ELBO = {elbo_after:.4f}, Change = {elbo_change:.4f}")
        
        # Check max iterations
        if self.iteration >= self.max_iterations:
            print(f"Reached maximum iterations ({self.max_iterations})")
            return True
            
        return elbo_change < self.convergence_threshold
    
    def get_convergence_stats(self) -> dict:
        """Return convergence statistics."""
        return {
            'final_elbo': self.elbo_history[-1],
            'num_iterations': self.iteration,
            'elbo_history': self.elbo_history,
            'parameter_changes': self.parameter_changes,
            'converged': self.iteration < self.max_iterations
        } 