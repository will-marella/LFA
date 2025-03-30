import numpy as np
from scipy.special import digamma, gammaln

class MFVIModel:
    def __init__(self, W: np.ndarray, alpha: np.ndarray, num_topics: int):
        """Initialize MFVI model parameters."""
        self.W = W
        self.M, self.D = W.shape
        self.num_topics = num_topics
        self.alpha = alpha
        
        # Initialize parameters
        params = self._initialize_parameters(self.M, self.D, num_topics, alpha_init=alpha)
        
        # Extract initialized values from params
        self.beta = params['beta_init']
        self.z = params['z_init']
        
        # Initial parameter updates
        self.e_theta = self._update_e_theta(self.alpha, self.z)
        self.e_log_theta = self._update_e_log_theta(self.e_theta)
        self.z = self._update_e_z(self.beta, self.W, self.e_log_theta)
        self.beta = self._update_beta(self.W, self.z)
        
    def _initialize_parameters(self, M, D, K, alpha_init=1):
        np.random.seed(42)
        alpha = np.full(K, alpha_init)
        beta_init = 0.5 + (np.random.rand(K, D) - 0.5) * 0.1  # Small perturbation around 0.5
        z = 1/K + (np.random.rand(M, D, K) - 0.5) * 0.1  # Small perturbation around 1/K
        z_init = z / np.sum(z, axis=2, keepdims=True) 
        
        params = {
            'alpha': alpha,
            'beta_init': beta_init,
            'z_init': z_init,
        }
        
        return params
        
    def update_parameters(self):
        """Perform one full iteration of parameter updates."""
        initial_elbo = self.compute_elbo(self.alpha, self.e_log_theta, self.e_theta, self.z, self.W, self.beta)
        previous_elbo = initial_elbo
        
        # Update e_theta
        self.e_theta = self._update_e_theta(self.alpha, self.z)
        elbo_after_e_theta = self.compute_elbo(self.alpha, self.e_log_theta, self.e_theta, self.z, self.W, self.beta)
        
        previous_elbo = elbo_after_e_theta
        
        # Update e_log_theta
        self.e_log_theta = self._update_e_log_theta(self.e_theta)
        elbo_after_e_log_theta = self.compute_elbo(self.alpha, self.e_log_theta, self.e_theta, self.z, self.W, self.beta)
        
        previous_elbo = elbo_after_e_log_theta
        
        # Update z
        self.z = self._update_e_z(self.beta, self.W, self.e_log_theta)
        elbo_after_e_z = self.compute_elbo(self.alpha, self.e_log_theta, self.e_theta, self.z, self.W, self.beta)
        
        previous_elbo = elbo_after_e_z
        
        # Update beta
        self.beta = self._update_beta(self.W, self.z)
        elbo_after_beta = self.compute_elbo(self.alpha, self.e_log_theta, self.e_theta, self.z, self.W, self.beta)
        
        param_changes = {
            'e_theta': abs(elbo_after_e_theta - initial_elbo),
            'z': abs(elbo_after_e_z - elbo_after_e_log_theta),
            'beta': abs(elbo_after_beta - elbo_after_e_z)
        }
        
        return initial_elbo, elbo_after_beta, param_changes

    def _update_e_theta(self, alpha, z):
        return alpha + np.sum(z, axis=1)

    def _update_e_log_theta(self, e_theta):
        return digamma(e_theta) - digamma(np.sum(e_theta, axis=1, keepdims=True))

    def _update_e_z(self, beta, W, e_log_theta):
        log_rho = np.log(beta + 1e-100)
        log_one_minus_rho = np.log(1 - beta + 1e-100)
        
        z = np.zeros((self.M, self.D, self.num_topics))
        for m in range(self.M):
            for d in range(self.D):
                log_phi = e_log_theta[m] + W[m,d] * log_rho[:,d] + (1 - W[m,d]) * log_one_minus_rho[:,d]
                log_phi = log_phi - np.max(log_phi)  # For numerical stability
                phi = np.exp(log_phi)
                z[m,d] = phi / np.sum(phi)
        return z

    def _update_beta(self, W, z):
        numerator = np.zeros((self.num_topics, self.D))
        denominator = np.zeros((self.num_topics, self.D))
        
        for m in range(self.M):
            numerator += np.multiply(z[m].T, W[m])
            denominator += z[m].T
            
        return numerator / (denominator + 1e-100)

    def compute_elbo(self, alpha, e_log_theta, e_theta, z, W, beta):
        """Compute the ELBO."""
        elbo = 0
        
        # E[log p(theta)]
        alpha_sum = np.sum(alpha)
        elbo += np.sum((alpha - 1) * e_log_theta)
        elbo += np.sum(gammaln(alpha))
        elbo -= np.sum(gammaln(alpha_sum))
        
        # E[log p(z|theta)]
        for m in range(self.M):
            elbo += np.sum(np.multiply(z[m], e_log_theta[m, np.newaxis]))
        
        # E[log p(W|z,beta)]
        log_rho = np.log(beta + 1e-100)
        log_one_minus_rho = np.log(1 - beta + 1e-100)
        for m in range(self.M):
            for d in range(self.D):
                elbo += np.sum(z[m,d] * (W[m,d] * log_rho[:,d] + (1 - W[m,d]) * log_one_minus_rho[:,d]))
        
        # H[q(z)]
        elbo += -np.sum(z * np.log(z + 1e-100))
        
        # H[q(theta)]
        e_theta_sum = np.sum(e_theta, axis=1)
        elbo += np.sum(gammaln(e_theta_sum))
        elbo -= np.sum(gammaln(e_theta))
        elbo += np.sum((e_theta - 1) * e_log_theta)
        
        return elbo 