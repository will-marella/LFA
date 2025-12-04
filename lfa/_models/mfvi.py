# Simulated dataset -- LFA via MFVI

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
np.euler_gamma  # Euler-Mascheroni constant

# Set random seed for reproducibility
np.random.seed(42)

## Form simulated dataset ###

# # Simple Parameters
# M = 50000  # Number of subjects
# D = 10
# num_topics_sim = 2
# alpha_sim = np.ones(num_topics_sim) / 1
# # beta = np.array([[0.9]*5 + [0.1]*5,
# #                  [0.1]*5 + [0.9]*5])
# beta = np.array([[0.5]*5 + [0.5]*5,
#                  [0.5]*5 + [0.5]*5])


# More complicated parameters
M = 10000  # Number of subjects
D = 20   # Number of diseases
num_topics_sim = 4   # Number of topics
alpha_sim = np.ones(num_topics_sim) / 10
# beta = np.array([
#     [0.9] * 5 + [0.1] * 15,  # Topic 1
#     [0.1] * 5 + [0.9] * 5 + [0.1] * 10,  # Topic 2
#     [0.1] * 10 + [0.9] * 5 + [0.1] * 5,  # Topic 3
#     [0.1] * 15 + [0.9] * 5   # Topic 4
# ])
beta = np.array([
    [0.8] * 5 + [0.2] * 15,  # Topic 1
    [0.2] * 5 + [0.8] * 5 + [0.2] * 10,  # Topic 2
    [0.2] * 10 + [0.8] * 5 + [0.2] * 5,  # Topic 3
    [0.2] * 15 + [0.8] * 5   # Topic 4
])
# beta = np.array([
#     [0.75] * 5 + [0.25] * 15,  # Topic 1
#     [0.25] * 5 + [0.75] * 5 + [0.25] * 10,  # Topic 2
#     [0.25] * 10 + [0.75] * 5 + [0.25] * 5,  # Topic 3
#     [0.25] * 15 + [0.75] * 5   # Topic 4
# ])
# beta = np.array([
#     [0.6] * 5 + [0.4] * 15,  # Topic 1
#     [0.4] * 5 + [0.6] * 5 + [0.4] * 10,  # Topic 2
#     [0.4] * 10 + [0.6] * 5 + [0.4] * 5,  # Topic 3
#     [0.4] * 15 + [0.6] * 5   # Topic 4
# ])
# beta = np.array([
#     [0.5] * 20,  # Topic 1
#     [0.5] * 20,  # Topic 2
#     [0.5] * 20,  # Topic 3
#     [0.5] * 20   # Topic 4
# ])
# beta = np.array([
#     [0.3] * 5 + [0.0] * 15,  # Topic 1
#     [0.0] * 5 + [0.3] * 5 + [0.0] * 10,  # Topic 2
#     [0.0] * 10 + [0.3] * 5 + [0.0] * 5,  # Topic 3
#     [0.0] * 15 + [0.3] * 5   # Topic 4
# ])




## Simulations

# Simulate the topic weights for each subject
theta = np.random.dirichlet(alpha_sim, M)

# print("theta simulation is:", theta)

# Simulate the observed diagnoses
W = np.zeros((M, D), dtype=np.int32)
for m in range(M):
    for d in range(D):
        # Topic assignment sampled from theta
        z_md = np.random.choice(num_topics_sim, p=theta[m])
        # Diagnosis sampled from beta
        W[m, d] = np.random.binomial(1, beta[z_md, d])


## Run the algorithm

def initialize_parameters(M, D, K, alpha_init=1):

    
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

####################################################################################
####################################################################################

def update_e_theta(alpha, z):
    e_theta = alpha + np.sum(z, axis=1)   
    return e_theta

def update_e_log_theta(e_theta):
    e_log_theta = digamma(e_theta) - digamma(np.sum(e_theta, axis=1, keepdims=True))
    return e_log_theta


def update_e_z(beta, W, e_log_theta, epsilon=1e-15):
    
    beta_safe = np.clip(beta, epsilon, 1 - epsilon)
    
    # log_beta = np.log(beta_safe).T[np.newaxis, :, :]  # Log of beta
    # log_one_minus_beta = np.log(1 - beta_safe).T[np.newaxis, :, :]  # Log of 1-beta
    
    log_beta = np.log(beta).T[np.newaxis, :, :]  # Log of beta
    log_one_minus_beta = np.log(1 - beta).T[np.newaxis, :, :]  # Log of 1-beta
    
    W_broadcasted = W[:, :, np.newaxis]  # Correctly shape (M, D, 1)
    
    log_prob_diseases_given_topics = (W_broadcasted * log_beta) + ((1 - W_broadcasted) * log_one_minus_beta)

    # Update E[z] using log-sum-exp trick for numerical stability
    e_log_theta_broadcasted = e_log_theta[:, np.newaxis, :]  # Add a new axis (M, 1, K)
    
    log_numerator = (e_log_theta_broadcasted + log_prob_diseases_given_topics) 
    
    # Subtract max for numerical stability before exponentiation (prevents overflow)
    log_numerator_stable = log_numerator - np.max(log_numerator, axis=2, keepdims=True)
    
    # Explicit exponentiation
    numerator = np.exp(log_numerator_stable)
    
    # Compute denominator (sum over K dimension)
    denominator = np.sum(numerator, axis=2, keepdims=True)
    
    # Compute final probabilities
    e_z = numerator / denominator
    
    return e_z


def update_beta(W, z, epsilon=1e-15):
    # Multiply the current expectation of z by W, sum over subjects (axis 0)
    numerator = np.sum(z * W[:, :, np.newaxis], axis=0)
    
    # Sum the current expectation of z over subjects (axis 0)
    denominator = np.sum(z, axis=0)

    safe_denominator = denominator + epsilon
    safe_numerator = numerator + epsilon

    # Calculate the new beta values
    updated_beta = (safe_numerator / safe_denominator).T  # Transpose to match expected dimension (topics x diseases)
    #
    
    ## Adding to try
    # Clip beta values to ensure they're in the range (epsilon, 1-epsilon)
    updated_beta = np.clip(updated_beta, epsilon, 1 - epsilon)
    
    
    return updated_beta


def compute_elbo(alpha, e_log_theta, e_theta, e_z, W, beta):
    
    # log_value_range(alpha, "Alpha for gammaln")
    # log_value_range(np.sum(alpha), "Sum of Alpha for gammaln")

    M, K = e_theta.shape
    D = W.shape[1]
    
    # Constants for clipping to avoid log(0)
    epsilon = 1e-15
    
    
    # Log probability of the prior
    log_prob_prior_term = np.sum(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(((alpha - 1) * e_log_theta), axis=1)) # Final scalar  ### 4 VO
    
    # Log probability of z given theta
    e_log_theta_expanded = e_log_theta[:, np.newaxis, :]  # Shape: (M, 1, K) ### 0 VO
    log_prob_z_theta_term = np.sum(e_z * e_log_theta_expanded)
    
    # print("log_prob_z_theta_term is: ", log_prob_z_theta_term)

    ## Log probability of the data
    safe_betaT = (np.clip(beta, epsilon, 1-epsilon)).T 
    beta_forUse = safe_betaT[np.newaxis, :, :]
    W_forUse = W[:, :, np.newaxis]
    
    log_prob_data_term = np.sum(e_z * ((W_forUse * np.log(beta_forUse)) + ((1-W_forUse)*(np.log(1-beta_forUse)))))
    
    # print("log_prob_data_term is: ", log_prob_data_term)

    # Entropy of theta
    entropy_theta = np.sum(gammaln(np.sum(e_theta, axis=1)) - np.sum((gammaln(e_theta)), axis=1) + np.sum(((e_theta - 1) * e_log_theta), axis=1))

    ## Entropy of z
    safe_e_z = np.clip(e_z, epsilon, 1-epsilon)
    entropy_z = -np.sum(e_z * np.log(safe_e_z))  ### 3 VO
    
    # print("entropy_z is: ", entropy_z)
    
    # log_value_range(alpha_sum_ez, "alpha_sum_ez for gammaln")
    # log_value_range(np.sum(alpha_sum_ez, axis=1), "Sum of alpha_sum_ez for gammaln")

    # Combine all the terms to compute the ELBO
    elbo = log_prob_prior_term + log_prob_z_theta_term + log_prob_data_term - entropy_theta + entropy_z  ### 4 VO
    
    # print("log_prob_prior_term is: " + str(log_prob_prior_term))
    # print("log_prob_z_theta_term is: " + str(log_prob_z_theta_term))
    # print("log_prob_data_term is: " + str(log_prob_data_term))
    # print("entropy_theta is: " + str(entropy_theta))
    # print("entropy_z is: " + str(entropy_z))
    
    # print(f"e_z shape: {e_z.shape}")
    # print(f"W_forUse shape: {W_forUse.shape}")
    # print(f"beta_forUse shape: {beta_forUse.shape}")
    # print(f"Result of W_forUse * np.log(beta_forUse) shape: {(W_forUse * np.log(beta_forUse)).shape}")
    # print(f"Full term before sum shape: {(e_z * ((W_forUse * np.log(beta_forUse)) + ((1-W_forUse)*(np.log(1-beta_forUse))))).shape}")

    return elbo



def perform_mfvi(W, alpha, num_topics, max_iterations, convergence_threshold):
    
    # Initialize parameters
    params = initialize_parameters(W.shape[0], W.shape[1], num_topics, alpha_init=alpha)
    
    # Extract initialized values from params
    beta = params['beta_init']
    z = params['z_init']
    
    # Initial ELBO computation
    e_theta = update_e_theta(alpha, z)
    e_log_theta = update_e_log_theta(e_theta)
    z = update_e_z(beta, W, e_log_theta)
    beta = update_beta(W, z)
    previous_elbo = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)

    negative_changes_e_theta = 0  # Counter for negative changes after updating e_theta
    negative_changes_e_log_theta = 0  # Counter for negative changes after updating e_log_theta
    negative_changes_e_z = 0  # Counter for negative changes after updating e_z
    negative_changes_beta = 0  # Counter for negative changes after updating beta
    total_negative_change_e_theta = 0.0  # Accumulator for total negative change after updating e_theta
    total_negative_change_e_log_theta = 0.0  # Accumulator for total negative change after updating e_log_theta
    total_negative_change_e_z = 0.0  # Accumulator for total negative change after updating e_z
    total_negative_change_beta = 0.0  # Accumulator for total negative change after updating beta

    # Begin iterative process
    for iteration in range(max_iterations):
        
        print(f'\nIteration {iteration}')
        
        initial_elbo = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)
        
        e_theta = update_e_theta(alpha, z)
        elbo_after_e_theta = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)
        change_e_theta = elbo_after_e_theta - previous_elbo
        print(f'After updating e_theta, ELBO = {elbo_after_e_theta:.4f}, Change after updating theta = {change_e_theta:.4f}')
        if change_e_theta < 0:
            negative_changes_e_theta += 1
            total_negative_change_e_theta += change_e_theta
        
        previous_elbo = elbo_after_e_theta
            
        e_log_theta = update_e_log_theta(e_theta)
        elbo_after_e_log_theta = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)
        change_e_log_theta = elbo_after_e_log_theta - previous_elbo
        print(f'After updating e_log_theta, ELBO = {elbo_after_e_log_theta:.4f}, Change after updating log theta = {change_e_log_theta:.4f}')
        if change_e_log_theta < 0:
            negative_changes_e_log_theta += 1
            total_negative_change_e_log_theta += change_e_log_theta 
            
        previous_elbo = elbo_after_e_log_theta
            
        z = update_e_z(beta, W, e_log_theta)
        elbo_after_e_z = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)
        change_e_z = elbo_after_e_z - previous_elbo
        print(f'After updating e_z, ELBO = {elbo_after_e_z:.4f}, Change after updating z = {change_e_z:.4f}')
        if change_e_z < 0:
            negative_changes_e_z += 1
            total_negative_change_e_z += change_e_z
            
        previous_elbo = elbo_after_e_z
        
        beta = update_beta(W, z)
        elbo_after_beta = compute_elbo(alpha, e_log_theta, e_theta, z, W, beta)
        change_beta = elbo_after_beta - previous_elbo
        print(f'After updating beta, ELBO = {elbo_after_beta:.4f}, Change from M-step = {change_beta:.4f}')
        if change_beta < 0:
            negative_changes_beta += 1
            total_negative_change_beta += change_beta
        
        previous_elbo = elbo_after_beta
        
        # track_gradients(z, W, e_log_theta, beta, compute_elbo)


        # Use the ELBO after M-step to check for convergence
        if np.abs(initial_elbo - previous_elbo) < convergence_threshold:
            print(f'Convergence reached. -- {negative_changes_e_theta} iterations after updating e_theta, {negative_changes_e_log_theta} iterations after updating e_log_theta, {negative_changes_e_z} iterations after updating z, and {negative_changes_beta} iterations after updating beta led to negative change in ELBO.')
            print(f'Total negative change after updating e_theta: {total_negative_change_e_theta}')
            print(f'Total negative change after updating e_logtheta: {total_negative_change_e_log_theta}')
            print(f'Total negative change after updating z: {total_negative_change_e_z}')
            print(f'Total negative change after updating beta: {total_negative_change_beta}')
            break
        
        
    else:
        print(f'Maximum iterations reached without convergence. -- {negative_changes_e_theta} iterations after updating e_theta, {negative_changes_e_log_theta} iterations after updating e_log_theta, {negative_changes_e_z} iterations after updating z, and {negative_changes_beta} iterations after updating beta led to negative change in ELBO.')
        print(f'Total negative change after updating e_theta: {total_negative_change_e_theta}')
        print(f'Total negative change after updating e_logtheta: {total_negative_change_e_log_theta}')
        print(f'Total negative change after updating z: {total_negative_change_e_z}')
        print(f'Total negative change after updating beta: {total_negative_change_beta}')

    e_theta_normalized = e_theta / np.sum(e_theta, axis=1, keepdims=True)

    # Preparing the result in a similar format to perform_cgs()
    combined_result = {
        'beta': beta,
        'z': z,  # Note: z here represents expected assignments, not hard assignments
        'theta': e_theta_normalized,
        'final_elbo': previous_elbo,  # Include the final ELBO in the results
        'num_iterations': iteration + 1
    }
    
    return combined_result





####################################################################################
####################################################################################

def calculate_correlation(beta1, beta2):
    # Add a small noise to avoid division by zero in std computation
    noise = np.random.normal(0, 0.001, beta1.shape)
    beta1_noisy = beta1 + noise
    beta2_noisy = beta2 + noise
    
    correlation_matrix = np.corrcoef(beta1_noisy, beta2_noisy)
    return correlation_matrix[:beta1.shape[0], beta1.shape[0]:]  # Ensure proper slicing

def align_topics_via_correlation(correlation_matrix):
    # Convert correlation to cost
    cost_matrix = 1 - correlation_matrix  # Higher correlation should mean lower cost
    # Use the Hungarian algorithm to find the best alignment based on correlation
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def align_topics(beta_estimated, beta_simulated, theta_estimated, theta_simulated):
    # Calculate the correlation matrix based on beta matrices
    correlation_matrix = calculate_correlation(beta_estimated, beta_simulated)

    # Align beta_estimated to beta_simulated using the Hungarian algorithm
    row_ind, col_ind = align_topics_via_correlation(correlation_matrix)
    
    # Reorder the estimated beta and theta matrices according to the optimal alignment
    aligned_beta_estimated = beta_estimated[col_ind, :]
    aligned_theta_estimated = theta_estimated[:, col_ind]
    
    return aligned_beta_estimated, beta_simulated, aligned_theta_estimated, theta_simulated

def mae(estimated, true):
    
    mae = np.mean(np.abs(estimated - true))
       
    return mae

def pearson_correlation(estimated, true):
    
    # Flatten arrays to make them 1D for correlation computation
    estimated_flat = estimated.flatten()
    true_flat = true.flatten()
    correlation, _ = pearsonr(estimated_flat, true_flat)
    
    return correlation

def compute_metrics(aligned_beta_estimated, aligned_beta_true, aligned_theta_estimated, aligned_theta_true):
    beta_mae_value = mae(aligned_beta_estimated, aligned_beta_true)
    beta_pearson_corr = pearson_correlation(aligned_beta_estimated, aligned_beta_true)
    
    theta_mae_value = mae(aligned_theta_estimated, aligned_theta_true)
    theta_pearson_corr = pearson_correlation(aligned_theta_estimated, aligned_theta_true)
    
    return beta_mae_value, beta_pearson_corr, theta_mae_value, theta_pearson_corr

def print_metrics(beta_mae_value, beta_pearson_corr, theta_mae_value, theta_pearson_corr):
    
    print("MAE between aligned beta matrices:", beta_mae_value)
    print("Pearson correlation between aligned beta matrices:", beta_pearson_corr)
    print("MAE between aligned theta matrices:", theta_mae_value)
    print("Pearson correlation between aligned theta matrices:", theta_pearson_corr)
    
    return

def beta_heatmap(aligned_beta_estimated):

    plt.figure(figsize=(10, 8))
    sns.heatmap(aligned_beta_estimated, annot=True, fmt=".2f", cmap='binary', cbar=True)
    plt.title('Heatmap of Aligned Beta Matrix from MFVI')
    plt.xlabel('Diseases')
    plt.ylabel('Topics')
    plt.show()
    
    return

def process_results(result, simulated_beta, simulated_theta):
    estimated_beta = result['beta']
    theta_estimated = result['theta']
    
    # Align the beta matrices
    aligned_beta_estimated, aligned_beta_true, aligned_theta_estimated, aligned_theta_true = align_topics(estimated_beta, simulated_beta, theta_estimated, simulated_theta)

    # Compute metrics
    beta_mae_value, beta_pearson_corr, theta_mae_value, theta_pearson_corr = compute_metrics(aligned_beta_estimated, aligned_beta_true, aligned_theta_estimated, aligned_theta_true)

    # Add these metrics to the result dictionary
    result['metrics'] = {
        'beta_mae': beta_mae_value,
        'beta_pearson_corr': beta_pearson_corr,
        'theta_mae': theta_mae_value,
        'theta_pearson_corr': theta_pearson_corr,
        'aligned_beta_estimated': aligned_beta_estimated
    }

    return result

def print_metrics(result):
    metrics = result['metrics']
    print("MAE between aligned beta matrices:", metrics['beta_mae'])
    print("Pearson correlation between aligned beta matrices:", metrics['beta_pearson_corr'])
    print("MAE between aligned theta matrices:", metrics['theta_mae'])
    print("Pearson correlation between aligned theta matrices:", metrics['theta_pearson_corr'])
    print("Number of iterations:", result['num_iterations'])
    
    
    

    
####################################################################################
####################################################################################


# Initialize parameters
num_topics = 4
alpha = np.ones(num_topics) / 10

# Set convergence threshold and max iterations
convergence_threshold = 0
max_iterations = 1000

# Perform MFVI
result = perform_mfvi(W, alpha, num_topics, max_iterations, convergence_threshold)

# Process results
result = process_results(result, beta, theta)

# Print out the results
# print("Aligned Estimated Beta Matrix:\n", result['metrics']['aligned_beta_estimated'])
# print_metrics(result)
beta_heatmap(result['metrics']['aligned_beta_estimated'])





