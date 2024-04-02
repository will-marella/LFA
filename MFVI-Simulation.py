# Simulated dataset -- LFA via MFVI

import numpy as np
from scipy.special import digamma, gammaln

# Set random seed for reproducibility
np.random.seed(42)

### Form simulated dataset ###

# Parameters
M = 1000  # Number of subjects
D = 10   # Number of diseases
K = 2   # Number of topics
alpha = np.array([1, 1])  # Dirichlet hyperparameter

# Topic loadings beta, rows correspond to topics and columns to diseases
beta = np.array([[0.9]*5 + [0.1]*5,
                 [0.1]*5 + [0.9]*5])

# Simulate the topic weights for each subject
theta = np.random.dirichlet(alpha, M)

# Simulate the observed diagnoses
W = np.zeros((M, D), dtype=np.int32)
for m in range(M):
    for d in range(D):
        # Assign topic based on theta
        z_md = np.random.choice(K, p=theta[m])
        # Sample diagnosis based on beta
        W[m, d] = np.random.binomial(1, beta[z_md, d])


def initialize_parameters(M, D, K, alpha_init=1):
    np.random.seed(42)  # For reproducibility
    
    # Dirichlet hyperparameter for topic weights
    alpha = np.full(K, alpha_init)
    
    # Initialize topic loadings, beta
    beta_init = 0.5 + (np.random.rand(K, D) - 0.5) * 0.1  # Small perturbation around 0.5

    
    # Initialize expected topic assignments, z
    z = 1/K + (np.random.rand(M, D, K) - 0.5) * 0.1  # Small perturbation around 1/K
    z_init = z / np.sum(z, axis=2, keepdims=True) 
    
    # Package parameters into a dictionary
    params = {
        'alpha': alpha,
        'beta_init': beta_init,
        'z_init': z_init,  # Add the initial expected topic assignments to the parameters
    }
    
    return params


def E_step(W, alpha, beta, z):
    
    """
    Perform the E-step of the variational inference algorithm.
    """

    # Calculation of expected theta values (e_theta) using broadcasting
    e_theta = alpha + np.sum(z, axis=1)

    # Calculate the expected log theta values (e_log_theta)
    e_log_theta = digamma(e_theta) - digamma(np.sum(e_theta, axis=1, keepdims=True))

    # Calculate log probabilities for diseases given topics
    beta_broadcasted = np.log(beta).T[np.newaxis, :, :]  # Correctly shape (1, D, K)
    W_broadcasted = W[:, :, np.newaxis]  # Correctly shape (M, D, 1)
    log_prob_diseases_given_topics = (W_broadcasted * beta_broadcasted) + ((1 - W_broadcasted) * np.log(1 - beta_broadcasted))

    # Update E[z]
    e_log_theta_broadcasted = e_log_theta[:, np.newaxis, :]  # Add a new axis (M, 1, K)
    numerator = np.exp(e_log_theta_broadcasted + log_prob_diseases_given_topics)
    denominator = np.sum(numerator, axis=2, keepdims=True)
    e_z = numerator / denominator
    
    return e_theta, e_log_theta, e_z


def M_step(W, z):
    
    """
    Perform the M-step of the variational inference algorithm.
    """
    
    # Multiply the current expectation of z by W, sum over subjects (axis 0)
    numerator = np.sum(z * W[:, :, np.newaxis], axis=0)
    
    # Sum the current expectation of z over subjects (axis 0)
    denominator = np.sum(z, axis=0)
    
    # Ensure that the denominator is not zero to avoid division by zero
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    
    # Calculate the new beta values
    updated_beta = (numerator / denominator).T
    
    return updated_beta


def compute_elbo(alpha, e_log_theta, e_theta, e_z, W, beta):
    
    """
    Compute the Evidence Lower Bound (ELBO) for the variational inference algorithm.
    """

    M, K = e_theta.shape
    D = W.shape[1]

    ## Log probability of the prior
    lpp_term1 = gammaln(np.sum(alpha))
    lpp_term2 = np.sum(gammaln(alpha))
    lpp_term3 = np.sum((alpha - 1) * e_log_theta)

    log_prob_prior_term = (M * lpp_term1) - (M * lpp_term2) + lpp_term3 # Final scalar

    ## Log probability z given theta
    e_log_theta_expanded = e_log_theta[:, np.newaxis, :]  # Shape: (M, 1, K)
    weighted_e_log_theta = e_log_theta_expanded * e_z  # Shape: (M, D, K)
    sum_weighted_e_log_theta = np.sum(weighted_e_log_theta, axis=1)  # Shape: (M, K)

    log_prob_z_theta_term = np.sum(sum_weighted_e_log_theta) # Final scalar

    ## Log probability of the data
    # Broadcasting W to have the same third dimension as z
    W_expanded = W[:, :, np.newaxis]  # Shape will be (M, D, 1)

    # Compute log terms
    log_beta = np.log(beta).T  # Shape will be (D, K)
    log_one_minus_beta = np.log(1 - beta).T  # Shape will be (D, K)

    # Broadcasting log terms to align with W and z
    log_beta_expanded = log_beta[np.newaxis, :, :]  # Adding subject dimension (1, D, K)
    log_one_minus_beta_expanded = log_one_minus_beta[np.newaxis, :, :]  # Adding subject dimension (1, D, K)

    # Weighted log probabilities for diseases given topics
    log_prob_terms = W_expanded * log_beta_expanded + (1 - W_expanded) * log_one_minus_beta_expanded  # (M, D, K)

    log_prob_data_term = np.sum(e_z * log_prob_terms)  # Scalar value

    ## Entropy of theta
    # Step 1: Sum E[z] over diseases (axis 1)
    sum_ez_over_diseases = np.sum(e_z, axis=1)  # Resulting shape (M, K)

    # Step 2: Add alpha (broadcasted over M) to sum of E[z]
    alpha_sum_ez = alpha[np.newaxis, :] + sum_ez_over_diseases  # Resulting shape (M, K)

    # Step 3: Apply gammaln function to alpha_sum_ez and alpha
    log_gamma_alpha_sum_ez = gammaln(alpha_sum_ez)  # Resulting shape (M, K)

    # Step 4: Calculate first term (sum over topics, sum over subjects)
    et_term_1 = np.sum(gammaln(np.sum(alpha_sum_ez, axis=1)))

    # Step 5: Calculate second term of the ELBO expression (sum over subjects and topics)
    et_term_2 = np.sum(log_gamma_alpha_sum_ez)

    # Step 6: Calculate third term (sum over subjects and topics after element-wise multiplication)
    et_term_3 = np.sum((alpha_sum_ez - 1) * e_log_theta)  # Scalar

    # Step 7: Combine
    entropy_theta = et_term_1 - et_term_2 + et_term_3

    ## Entropy of z
    entropy_z = np.sum(e_z * np.log(e_z))  # Entropy term for z

    # Combine all the terms to compute the ELBO
    elbo = log_prob_prior_term + log_prob_z_theta_term + log_prob_data_term - entropy_theta - entropy_z

    return elbo


###### Run Model

# Initialize parameters
params = initialize_parameters(M, D, K)
alpha = params['alpha']
beta = params['beta_init']
z = params['z_init']

# Set convergence threshold and max iterations
convergence_threshold = 1e-4
max_iterations = 1000
previous_elbo = np.inf

# Begin iterative process
for iteration in range(max_iterations):
    # E-step: Update the expectation of z and related quantities
    e_theta, e_log_theta, z = E_step(W, alpha, beta, z)
    # print("Z is:" + str(z))
    
    # M-step: Update the parameter beta
    beta = M_step(W, z)
    
    # Compute the ELBO with the newly updated parameters
    elbo_value = compute_elbo(params['alpha'], e_log_theta, e_theta, z, W, beta)
    
    # Print the ELBO value to monitor convergence
    print(f'Iteration {iteration}: ELBO = {elbo_value:.4f}, Change = {elbo_value - previous_elbo:.4f}')

    # Check for convergence by comparing the change in ELBO to the threshold
    if np.abs(previous_elbo - elbo_value) < convergence_threshold:
        print('Convergence reached.')
        break
    previous_elbo = elbo_value
else:
    print('Maximum iterations reached without convergence.')

print(f'Final ELBO: {elbo_value:.4f}')


# Print final parameters
print("Data:\n", W)
print("Final theta values:\n", e_theta)
print("Final beta values:\n", beta)
print("Final z values:\n", z)