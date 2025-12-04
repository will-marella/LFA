import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


def calculate_correlation(beta1, beta2):
    
    # Calculate the correlation matrix between two sets of beta distributions
    correlation_matrix = np.corrcoef(beta1, beta2)[:len(beta1), len(beta1):]
    
    return 1 - correlation_matrix  # Convert to distance matrix for alignment

def align_topics_via_correlation(correlation_matrix):
    
    # Use the Hungarian algorithm to find the best alignment based on correlation
    row_ind, col_ind = linear_sum_assignment(correlation_matrix)
    
    return row_ind, col_ind

def align_chains(results):
    num_chains = len(results)
    reference_chain_index = 0  # Use the first chain as reference
    
    print(f"Number of beta samples in reference chain: {len(results[reference_chain_index]['beta_samples'])}")
    print(f"Shape of first beta sample: {results[reference_chain_index]['beta_samples'][0].shape}")
    
    reference_beta = np.mean(np.array(results[reference_chain_index]['beta_samples']), axis=0)
    
    print(f"Shape of reference_beta: {reference_beta.shape}")

    aligned_results = []

    for i in range(num_chains):
        if i == reference_chain_index:
            aligned_results.append(results[i])  # Append reference output directly
            continue

        current_beta_avg = np.mean(np.array(results[i]['beta_samples']), axis=0)
        correlation_matrix = calculate_correlation(reference_beta, current_beta_avg)

        _, col_ind = align_topics_via_correlation(correlation_matrix)

        aligned_beta_samples = [beta[col_ind, :] for beta in results[i]['beta_samples']]

        # Correctly align z_samples
        aligned_z_samples = []
        for z_sample in results[i]['z_samples']:
            aligned_z = np.array([[col_ind[z] for z in row] for row in z_sample])
            aligned_z_samples.append(aligned_z)

        # Correctly align theta_samples (permute topic columns)
        aligned_theta_samples = [theta[:, col_ind] for theta in results[i]['theta_samples']]

        aligned_results.append({
            'beta_samples': aligned_beta_samples,
            'z_samples': aligned_z_samples,
            'theta_samples': aligned_theta_samples
        })

    return aligned_results

def average_aligned_BetasAndThetas(aligned_results):
    # Extract all aligned beta and theta matrices
    all_aligned_beta = np.array([res['beta_samples'] for res in aligned_results])
    all_aligned_theta = np.array([res['theta_samples'] for res in aligned_results])
    
    # Average beta and theta results across the chains
    mean_beta = np.mean(all_aligned_beta, axis=(0, 1))  # Averaging over both chains and iterations
    mean_theta = np.mean(all_aligned_theta, axis=(0, 1))  # Averaging over both chains and iterations
    

    return mean_beta, mean_theta



def calculate_z_distribution(aligned_results):
    
    # Concatenate all z_samples across all chains
    concatenated_z = np.concatenate([res['z_samples'] for res in aligned_results], axis=0)

    # Calculate the frequency of each topic assignment for each subject-disease pair
    M, D = concatenated_z[0].shape  # Assuming all z_samples have the same shape
    num_topics = aligned_results[0]['beta_samples'][0].shape[0]  # Assuming shape consistency in beta samples

    topic_counts = np.zeros((M, D, num_topics), dtype=int)

    # Populate the count matrix
    for sample in concatenated_z:
        for m in range(M):
            for d in range(D):
                topic_counts[m, d, sample[m, d]] += 1

    # Convert counts to probabilities
    total_samples = len(concatenated_z)  # Now directly using the length of concatenated_z
    z_distribution = topic_counts / total_samples

    return z_distribution


def merge_chains(aligned_results):
    # Calculate z distribution from the aligned results
    z_distribution = calculate_z_distribution(aligned_results)

    # Calculate averaged beta and theta from aligned results
    mean_beta, mean_theta = average_aligned_BetasAndThetas(aligned_results)

    combined_result = {
        'z_distribution': z_distribution,
        'beta': mean_beta,
        'theta': mean_theta
    }

    return combined_result

def calculate_correlation_2(beta1, beta2):
    # Transpose beta matrices to correlate topics (rows) instead of diseases (columns)
    correlation_matrix = np.corrcoef(beta1, beta2)[:len(beta1), len(beta1):]

    # Replace any NaN or infinite correlations (which arise when a vector has zero variance)
    # Treat undefined correlations as 0 (no linear relationship)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    return correlation_matrix

def align_to_simulated_topics(combined_result, simulated_beta):
    
    estimated_beta = combined_result['beta']
    
    # Calculate the correlation matrix
    correlation_matrix = calculate_correlation_2(estimated_beta, simulated_beta)
    cost_matrix = -correlation_matrix

    # Align estimated_beta to simulated_beta using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the topic mapping
    topic_mapping = {est: sim for est, sim in zip(row_ind, col_ind)}
    
    # Create a reverse mapping to reorder the matrices
    reorder_mapping = {sim: est for est, sim in topic_mapping.items()}
    reorder_indices = [reorder_mapping[i] for i in range(len(reorder_mapping))]
    
    # Reorder the matrices
    combined_result['beta'] = combined_result['beta'][reorder_indices, :]
    combined_result['theta'] = combined_result['theta'][:, reorder_indices] 
    combined_result['z_distribution'] = combined_result['z_distribution'][:, :, reorder_indices]

    return combined_result

def mae(estimated, true):
    
    mae = np.mean(np.abs(estimated - true))
       
    return mae

def pearson_correlation(estimated, true):
    
    # Flatten arrays to make them 1D for correlation computation
    estimated_flat = estimated.flatten()
    true_flat = true.flatten()
    correlation, _ = pearsonr(estimated_flat, true_flat)
    
    return correlation

def compute_cgs_metrics(combined_result, simulated_beta, simulated_theta):
    estimated_beta = combined_result['beta']
    estimated_theta = combined_result['theta']

    beta_mae_value = mae(estimated_beta, simulated_beta)
    beta_pearson_corr = pearson_correlation(estimated_beta, simulated_beta)
    
    theta_mae_value = mae(estimated_theta, simulated_theta)
    theta_pearson_corr = pearson_correlation(estimated_theta, simulated_theta)

    metrics = {
        'beta_mae': beta_mae_value,
        'beta_pearson_corr': beta_pearson_corr,
        'theta_mae': theta_mae_value,
        'theta_pearson_corr': theta_pearson_corr
    }

    return metrics

def print_metrics(metrics):
    print("MAE between aligned beta matrices:", metrics['beta_mae'])
    print("Pearson correlation between aligned beta matrices:", metrics['beta_pearson_corr'])
    print("MAE between aligned theta matrices:", metrics['theta_mae'])
    print("Pearson correlation between aligned theta matrices:", metrics['theta_pearson_corr'])
    print("Run-time:", metrics['run_time'], "seconds")

def compute_mfvi_metrics(result, beta, theta):
    """
    Compute metrics specific to MFVI results.
    
    Args:
        result: Dictionary containing MFVI results
            - beta: Estimated topic-disease matrix
            - theta: Estimated topic distributions
            - z: Expected topic assignments
            - final_elbo: Final ELBO value
            - num_iterations: Number of iterations until convergence
        beta: True topic-disease matrix
        theta: True topic distributions
    """
    metrics = {}
    
    # Basic convergence metrics
    metrics['num_iterations'] = result['num_iterations']
    metrics['final_elbo'] = result['final_elbo']
    
    # Topic recovery metrics
    # Beta matrix correlation
    beta_corr = np.corrcoef(result['beta'].flatten(), beta.flatten())[0,1]
    metrics['beta_correlation'] = beta_corr
    
    # Topic distribution correlation
    theta_corr = np.corrcoef(result['theta'].flatten(), theta.flatten())[0,1]
    metrics['theta_correlation'] = theta_corr
    
    # Mean squared errors
    metrics['beta_mse'] = np.mean((result['beta'] - beta) ** 2)
    metrics['theta_mse'] = np.mean((result['theta'] - theta) ** 2)
    
    return metrics

def align_mfvi_results(result, simulated_beta):
    """Align MFVI results with simulated topics."""
    estimated_beta = result['beta']
    
    # Calculate the correlation matrix
    correlation_matrix = calculate_correlation_2(estimated_beta, simulated_beta)
    cost_matrix = -correlation_matrix

    # Align estimated_beta to simulated_beta using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the topic mapping
    topic_mapping = {est: sim for est, sim in zip(row_ind, col_ind)}
    
    # Create a reverse mapping to reorder the matrices
    reorder_mapping = {sim: est for est, sim in topic_mapping.items()}
    reorder_indices = [reorder_mapping[i] for i in range(len(reorder_mapping))]
    
    # Reorder the matrices
    result['beta'] = result['beta'][reorder_indices, :]
    result['theta'] = result['theta'][:, reorder_indices]
    if 'z' in result:  # Handle probabilistic z assignments if present
        result['z'] = result['z'][:, :, reorder_indices]

    return result

def print_mfvi_metrics(metrics):
    """Print metrics specific to MFVI results."""
    print("Number of iterations:", metrics['num_iterations'])
    print("Final ELBO:", metrics['final_elbo'])
    print("Beta correlation:", metrics['beta_correlation'])
    print("Theta correlation:", metrics['theta_correlation'])
    print("Beta MSE:", metrics['beta_mse'])
    print("Theta MSE:", metrics['theta_mse'])
    if metrics.get('mean_elbo_delta_tail') is not None:
        print("Mean |ΔELBO| (tail):", metrics['mean_elbo_delta_tail'])
    print("Run-time:", metrics['run_time'], "seconds")
