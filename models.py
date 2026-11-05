import torch
import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self, K: int, D: int):
        """Initialize the Naive Bayes model with K classes and D features.
        
        Args:
            K (int): Number of classes.
            D (int): Number of features.
        """
        self.K = K
        self.D = D
        self.feature_means = torch.zeros((K, D))  # Mean of each feature for each class
        self.feature_variances = torch.zeros((K, D))  # Variance of each feature for each class
        self.class_priors = torch.zeros(K)  # Prior probability of each class
        self.classes = None  # Unique classes in the dataset

    def _log_probability(self, X):
        """Helper function to compute log P(x|y==k) for all classes."""
        X_exp = X.unsqueeze(1)                    # size (N, 1, D)
        mean = self.feature_means.unsqueeze(0)              # size (1, K, D)
        var = self.feature_variances.unsqueeze(0)         # size (1, K, D)

        # Definition of Log Gaussian Probability: -0.5 * (log(2*pi*var) + (x-mean)^2/var)
        log_probs = -0.5 * (torch.log(2 * torch.pi * var) + (X_exp - mean)**2 / var)

        # Sum all feature probaility logs to apply independence assumption -> log addition is equivalent to multiplying probabilities
        return log_probs.sum(dim=2)  # size (N, K)

    def fit(self, X, y):
        """Fit features to Gaussian models, estimate class proportions with counts.
        
        Args:
            X (DataFrame): Feature matrix of shape (N, D) where N is the number of samples and D is the number of features. Must be continuous features, categorical features are not supported.
            y (DataFrame): Class string labels of shape (N,), with K unique classes.
        """
        if not (X.shape[0] == y.shape[0]):
            raise ValueError("Number of samples in X and y must be the same.")
        if not (X.shape[1] == self.D):
            raise ValueError(f"Number of features in X must be {self.D}.")
        if not (len(np.unique(y)) == self.K):
            raise ValueError(f"Number of unique classes in y must be {self.K}.")
        
        X = torch.tensor(X.values).float()
        self.classes = np.unique(y)

        # For each class, compute the mean and variance of the features, and the class prior (proportion of samples in that class)
        for k in range(self.K):
            X_k = X[y == self.classes[k]]  # Select samples belonging to class k
            self.feature_means[k] = X_k.mean(dim=0)
            self.feature_variances[k] = X_k.var(dim=0, correction=0) + 1e-4  # Add a small variance floor to avoid division by zero, also divide by N instead of N-1 for MLE estimate
            self.class_priors[k] = len(X_k) / len(X)
        return self

    def predict_labels(self, X_prime):
        """Predict class labels for new data X_prime. Must perform after fitting."""
        X_prime = torch.tensor(X_prime.values).float()
        # Previous probabilities of each class are used as priors, the starting point for the probability calculation
        log_prior = torch.log(self.class_priors)
        log_posterior = self._log_probability(X_prime) + log_prior

        # Apply argmax to get the class with the highest posterior probability for each sample
        argmax_indices = log_posterior.argmax(dim=1).numpy()
        return self.classes[argmax_indices]  # Return the predicted class labels
