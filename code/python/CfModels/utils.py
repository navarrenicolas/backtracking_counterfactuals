import numpy as np
from itertools import product

def binary_softmax(p, temp, method='exponential'):
    """
    Numerically stable and vectorized binary softmax function.
    
    Args:
        p (float or array): Probability value(s) between 0 and 1
        temp (float): Temperature parameter
        method (str): 'exponential' or 'power'
    
    Returns:
        float or array: Softmax-transformed probability
    """
    # Ensure inputs are arrays for vectorization
    p = np.asarray(p)
    scalar_input = p.ndim == 0
    p = np.atleast_1d(p)
    
    epsilon = 1e-9
    
    if method == 'power':
        # Log-sum-exp version of power method
        log_p_scaled = np.log(np.clip(p, epsilon, 1-epsilon)) / temp
        log_1_p_scaled = np.log(np.clip(1 - p, epsilon, 1-epsilon)) / temp
        
        # Log-sum-exp trick for numerical stability
        max_vals = np.maximum(log_p_scaled, log_1_p_scaled)
        exp_sum = np.exp(log_p_scaled - max_vals) + np.exp(log_1_p_scaled - max_vals)
        result = np.exp(log_p_scaled - max_vals) / exp_sum
        
    else:  # exponential method
        # Log-sum-exp version of exponential method
        log_p_scaled = p / temp
        log_1_p_scaled = (1 - p) / temp
        
        # Log-sum-exp trick for numerical stability
        max_vals = np.maximum(log_p_scaled, log_1_p_scaled)
        exp_sum = np.exp(log_p_scaled - max_vals) + np.exp(log_1_p_scaled - max_vals)
        result = np.exp(log_p_scaled - max_vals) / exp_sum
    
    
    # Return scalar if input was scalar
    if scalar_input:
        return result.item()
    return result

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))