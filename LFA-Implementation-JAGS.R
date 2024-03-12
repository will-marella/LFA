### Load packages
library(rjags)
library(gtools)
library(coda)
library(dplyr)
library(tidyr)
library(bayesplot)
library(tidybayes)
library(posterior)

################################################
### Simulate data

# Set seed for reproducibility
set.seed(42)

# Parameters
M <- 500  # Number of subjects
D <- 10   # Number of diseases
K <- 2    # Number of topics
alpha <- c(1, 1)  # Dirichlet hyperparameter


# Topic loadings beta, rows correspond to topics and columns to diseases
beta <- matrix(c(rep(0.9, 5), rep(0.1, 5), rep(0.1, 5), rep(0.9, 5)), nrow = K, byrow = TRUE)

# Simulate the topic weights for each subject based on the Dirichlet distribution
theta <- rdirichlet(M, alpha)

# Simulate the observed diagnoses
W <- matrix(nrow = M, ncol = D)

# Initialize a matrix to store actual topic assignments (actual_z)
actual_z <- matrix(nrow = M, ncol = D)

for (m in 1:M) {
  for (d in 1:D) {
    # Assign topic based on theta
    z_md <- sample(1:K, size = 1, prob = theta[m, ])
    # Store the assigned topic in actual_z matrix
    actual_z[m, d] <- z_md
    
    # Sample diagnosis based on beta
    W[m, d] <- rbinom(1, size = 1, prob = beta[z_md, d])
  }
}

# Put all into data for JAGS
jags_data <- list(M = M, D = D, K = K, W = W)

################################################
### Load the model
model_string <- "
model {
    # Priors for theta and beta
    for (m in 1:M) {
        theta[m, 1:K] ~ ddirch(alpha[])
    }
    for (k in 1:K) {
        for (d in 1:D) {
            beta[k, d] ~ dbeta(1, 1)  # Assuming a Beta(1, 1) prior for simplicity
        }
    }
    # Likelihood
    for (m in 1:M) {
        for (d in 1:D) {
            z[m, d] ~ dcat(theta[m, 1:K])
            W[m, d] ~ dbern(beta[z[m, d], d])
        }
    }
}
"

# Define parameters, number of chains
params <- c("theta", "beta", "z")
nchains <- 4

# Fit model
jags_model <- jags.model(textConnection(model_string), data = jags_data, n.chains = nchains, n.adapt = 1000)

# Run the model (burn-in)
update(jags_model, 10000)  # Adjust the burn-in period as needed

# Run the model (sampling)
mcmc_results <- coda.samples(model = jags_model, variable.names = params, n.iter = 5000)
  # Adjusting n.iterations as needed...

# Take draws from the posterior
draws <- as_draws(mcmc_results)

# Summarize the draws
summ <- summary(draws) 
as.data.frame(summ)
  # This summary is actually quite good and will be quite easy to work with

################################################
## Compare inferred z to actual z

# Filter rows for 'z' variables
z_summ <- summ %>%
  filter(grepl("^z\\[", variable)) %>%
  
  # Extract indices from 'variable' strings
  mutate(subject = as.integer(gsub("z\\[(\\d+),.*", "\\1", variable)),
         disease = as.integer(gsub("z\\[\\d+,(\\d+)\\]", "\\1", variable))) %>%
  
  # Select relevant columns and arrange by subject and disease
  select(subject, disease, median) %>%
  arrange(subject, disease)

# Put together inferred_z
inferred_z <- matrix(round(z_summ$median), nrow = max(z_summ$subject), ncol = max(z_summ$disease), byrow = TRUE)

# Calculate accuracy
accuracy <- sum(actual_z == inferred_z) / length(actual_z)
print(accuracy) # 0.107
  # So low! Suggests that the topic assignments were inverted

# Invert the inferred topic assignments
inferred_z_inverted <- 3 - inferred_z

# Recalculate accuracy with the inverted assignments
accuracy_inverted <- sum(actual_z == inferred_z_inverted) / length(actual_z)
print(accuracy_inverted)
  # 0.8926 for the topic assignments seems quite good






