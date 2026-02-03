import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ...existing code...
import sys
import os

# Ensure we can see the CfModels package (since notebook is in a subdir)
sys.path.append(os.path.abspath('../'))

agency_data = pd.read_csv('../../../data/agency/agency_cleaned.csv')

from scipy.optimize import minimize

from CfModels.utils import *
from CfModels.inference import CounterfactualInferenceBN
from CfModels.networks import create_diamond_bayesian_network

def convert_data(row):

    resps = {'A': row.A, 'C': row.C, 'D': row.D}
     
    order_vars = [c for c in row.order]
    path = [{v: resps[v]} for v in order_vars]
    
    return path

def extract_for_fit(row):
    path = convert_data(row)
    agent_C = row.agent_path
    agent_B = row.agent_intervention
    disjunction = True
    question = False
    fit_dict = {
        'path': path,
        'disjunction': disjunction,
        'question': question,
        'agent_C': agent_B,
        'agent_B': agent_C
    }
    return fit_dict


def unique_pattern_analysis(df,verbose = False):
    
    # Generate the list of extracted data
    data_list = [extract_for_fit(row) for _, row in df.iterrows()]

    # Create a manual counter using a dictionary
    pattern_counts = {}

    for item in data_list:
        # Convert the path list and other elements to a string representation for comparison
        path_str = str([(list(d.keys())[0], d[list(d.keys())[0]]) for d in item['path']])
        pattern_key = (path_str, item['disjunction'], item['question'], item['agent_B'], item['agent_C'])
        
        if pattern_key in pattern_counts:
            pattern_counts[pattern_key] += 1
        else:
            pattern_counts[pattern_key] = 1

    # Count how many items appear more than once
    repeated_count = sum(1 for count in pattern_counts.values() if count > 1)
    total_repeated_instances = sum(count for count in pattern_counts.values() if count > 1)

    if verbose:
        print(f"Number of unique patterns that are repeated: {repeated_count}")
        print(f"Total number of repeated instances: {total_repeated_instances}")
        print(f"Total instances: {len(data_list)}")

    # Show the most common patterns (sorted manually)
    pattern_list = [(pattern, count) for pattern, count in pattern_counts.items()]
    # Sort by count in descending order
    pattern_list.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print("\nMost common patterns:")
        for pattern, count in pattern_list[:10]:
            if count > 1:
                print(f"Count: {count}, Pattern: {pattern}")
    return pattern_list


def _params_from_reduced_unconstrained(u_reduced, param_names, fixed_settings):
    """
    Build full transformed parameters (s, p_keep, br, beta_A, temperature)
    from a reduced unconstrained vector u_reduced and any fixed settings.

    fixed_settings may contain any of: 's','p_keep','br','beta_A','temperature'
    If a parameter is fixed its (transformed) value is taken from fixed_settings.
    Otherwise values are read in order from u_reduced and transformed as in
    _unconstrained_to_params.
    """
    # Number of free params expected
    free_count = sum(1 for name in param_names if name not in fixed_settings)
    if len(u_reduced) != free_count:
        raise ValueError(f"Length of unconstrained vector ({len(u_reduced)}) "
                         f"does not match number of free parameters ({free_count}).")

    it = iter(u_reduced)
    # transformer functions for each unconstrained value

    out = {}
    for name in param_names:
        if name in fixed_settings:
            out[name] = fixed_settings[name]
        else:
            x = next(it)
            if name == 'temperature':
                out['temperature'] = np.exp(x)
            else:
                out[name] = sigmoid(x)
    return out


def compute_independent_baseline_nll(unique_patterns):
    """
    Computes the negative log-likelihood of a baseline model that assumes
    independence between answers and uses the empirical mean of each variable
    as the prediction probability.
    """
    # Calculate empirical means for A, C, D

    a_count= 0
    c_count = 0
    d_count = 0
    pattern_total = sum([count for _, count in unique_patterns])
    for pattern, count in unique_patterns:
        path = [dict([i]) for i in eval(pattern[0])]
        for step in path:
            if step.get('A', None) == 1:
                a_count += count
            if step.get('C', None) == 1:
                c_count += count
            if step.get('D', None) == 1:
                d_count += count
    mean_A = a_count / pattern_total
    mean_C = c_count / pattern_total
    mean_D = d_count / pattern_total
    means = {
        'A': mean_A,
        'C': mean_C,
        'D': mean_D
    }
    

    for pattern, count in unique_patterns:
        # pattern[0] contains the path as a string of list of tuples, e.g. "[('A', 1), ('C', 0)...]"
        # We parse it back into a list of tuples
        path = [dict([i]) for i in eval(pattern[0])]
        
        # Calculate joint probability of this specific path under independence assumption
        path_prob = 1.0
        for path_item in path:
             var, obs_val = list(path_item.items())[0]
             if var in means:
                 p_1 = means[var]
                 # Probability of the observed discrete value given the mean probability
                 # If obs is 1, likelihood is p; if obs is 0, likelihood is 1-p
                 if obs_val == 1:
                     p_obs = p_1
                 else:
                     p_obs = 1.0 - p_1
        
        ll += np.log(path_prob) * count
        
    return -ll

def negative_log_likelihood(params, unique_patterns, inference_method, param_names, fixed_settings={}, random_pred = False, verbose=False):
    """
    Now 'params' are unconstrained optimizer variables (real numbers) for the free parameters.
    They are transformed to the model domain here. fixed_settings can contain any subset of
    the model parameters (s, p_keep, br, beta_A, temperature) to hold fixed.
    """

    if random_pred:
            return compute_independent_baseline_nll(unique_patterns)
        
    try:
        transformed = _params_from_reduced_unconstrained(params, param_names, fixed_settings)
    except Exception as e:
        if verbose:
            print("Parameter reconstruction error:", e)
        return np.inf


    s = transformed['s']
    p_keep = transformed['p_keep']
    temperature = transformed['temperature']

    beta_A = transformed['beta_A']
    beta_D = transformed['beta_D']

    
    br_agent = transformed.get('br_agent', None)
    br_non_agent = transformed.get('br_non_agent', None)
    br = transformed.get('br', None)
    if br_agent is None and br_non_agent is None:
        if br is None:
            raise ValueError("br parameter cannot be None when br_agent and br_non_agent are not provided")
        br_agent = br
        br_non_agent = br
        
    cp_agent_source = transformed.get('cp_agent_source', None)
    cp_non_agent_source = transformed.get('cp_non_agent_source', None)
    cp_source = transformed.get('cp_source', None)
    if cp_agent_source is None and cp_non_agent_source is None:
        if cp_source is None:
            cp_agent_source = 1 - br_agent
            cp_non_agent_source = 1 - br_non_agent
        else:
            cp_agent_source = cp_source
            cp_non_agent_source = cp_source

    cp_agent_target = transformed.get('cp_agent_target', None)
    cp_non_agent_target = transformed.get('cp_non_agent_target', None)
    cp_target = transformed.get('cp_target', None)
    if cp_agent_target is None and cp_non_agent_target is None:
        if cp_target is None:
            cp_agent_target = 1 - br_agent
            cp_non_agent_target = 1 - br_non_agent
        else:
            cp_agent_target = cp_target
            cp_non_agent_target = cp_target

    # Build models with current parameters (command/outcome powers and base rates may be in fixed_settings)
    disj_model_b1_c1 = create_diamond_bayesian_network(
        power_B=cp_agent_source,
        power_C=cp_agent_source,
        power_D_from_B=cp_agent_target,
        power_D_from_C=cp_agent_target,
        beta_A=beta_A,
        beta_B=br_agent,
        beta_C=br_agent,
        beta_D=beta_D,
        disjunction= True
    )

    disj_model_b1_c0 = create_diamond_bayesian_network(
        power_B=cp_agent_source,
        power_C=cp_non_agent_source,
        power_D_from_B=cp_agent_target,
        power_D_from_C=cp_non_agent_target,
        beta_A=beta_A,
        beta_B=br_agent,
        beta_C=br_non_agent,
        beta_D=beta_D,
        disjunction= True
    )

    disj_model_b0_c1 = create_diamond_bayesian_network(
        power_B=cp_non_agent_source,
        power_C=cp_agent_source,
        power_D_from_B=cp_non_agent_target,
        power_D_from_C=cp_agent_target,
        beta_A=beta_A,
        beta_B=br_non_agent,
        beta_C=br_agent,
        beta_D=beta_D,
        disjunction= True
    )

    disj_model_b0_c0 = create_diamond_bayesian_network(
        power_B=cp_non_agent_source,
        power_C=cp_non_agent_source,
        power_D_from_B=cp_non_agent_target,
        power_D_from_C=cp_non_agent_target,
        beta_A=beta_A,
        beta_B=br_non_agent,
        beta_C=br_non_agent,
        beta_D=beta_D,
        disjunction= True
    )



    inference_engines = {
        (1,1): CounterfactualInferenceBN(disj_model_b1_c1),
        (1,0): CounterfactualInferenceBN(disj_model_b1_c0),
        (0,1): CounterfactualInferenceBN(disj_model_b0_c1),
        (0,0): CounterfactualInferenceBN(disj_model_b0_c0),
    } 



    # Compute log-likelihood (sum over pattern counts)
    ll = 0.0

    for pattern, count in unique_patterns:        

        path = [dict([i]) for i in eval(pattern[0])]
        disjunction = pattern[1]
        question = pattern[2]
        
        agent_combo = (pattern[3], pattern[4])
        
        

        cf_constraint = {'B':1} if question else {'B':0}
        obs = {'A':0,'B':0,'C':0,'D':0} if question else {'A':1,'B':1,'C':1,'D':1}
        
        result = inference_method(
            base_inf= inference_engines[agent_combo],
            disjunction=disjunction,
            path=path,
            obs=obs,
            cf_constraint=cf_constraint,
            s=s,
            temperature=temperature,
            p_keep=p_keep,
            verbose=False
        )
        prob =  result['probability']
        log_prob = np.log(prob)
        if np.isnan(log_prob) or np.isinf(log_prob) or prob <= 0:
            print(f"Invalid probability: {prob}, log_prob: {log_prob} for pattern: {pattern}")
            return np.inf
        # print(prob)
        ll += log_prob * count

    # return negative log-likelihood 
    return -ll

def fit_model(df, inference_method, param_names=None, fixed_settings={}, method='BFGS', verbose=True):
    """
    Use unconstrained optimization variables for only the free parameters.
    fixed_settings may include any of: 's','p_keep','br','beta_A','temperature' to hold them fixed.
    """
    # Allowed parameter names and default ordering for unconstrained vector
    if param_names is None:
        param_names = ['s', 'p_keep', 'br', 'beta_A', 'temperature']

    # Determine free parameters
    free_param_names = [n for n in param_names if n not in fixed_settings]
    n_free = len(free_param_names)

    if verbose:
        print(f"Fitting parameters. Fixed: { {k:v for k,v in fixed_settings.items() if k in param_names} }")
        print(f"Free parameters to optimize (in order): {free_param_names}")

    # initialize unconstrained free params (standard normal)
    if n_free > 0:
        initial_u = np.random.randn(n_free)
    else:
        initial_u = np.array([])

    unique_patterns = unique_pattern_analysis(df, verbose=False)

    if verbose:
        print(f"Initial unconstrained params (free only): {initial_u}")
        try:
            cur_nll = negative_log_likelihood(initial_u, unique_patterns, inference_method, param_names, fixed_settings)
            print(f"Initial NLL: {cur_nll}")
        except Exception as e:
            print("Initial NLL computation failed:", e)

    # If no free parameters, skip optimization and evaluate directly
    if n_free == 0:
        final_nll = negative_log_likelihood(initial_u, unique_patterns, inference_method, param_names, fixed_settings)
        fitted_params = _params_from_reduced_unconstrained(initial_u, param_names, fixed_settings)
        optimization_result = {
            'x': np.array([]),
            'fun': final_nll,
            'success': True,
            'message': 'No free parameters to optimize (all fixed via fixed_settings).'
        }
        if verbose:
            print("\nOptimization skipped (all parameters fixed).")
            print(f"Fitted params (transformed): {fitted_params}")
            print(f"Final NLL: {final_nll:.4f}")
        return {
            'fitted_params': fitted_params,
            'nll': final_nll,
            'success': True,
            'message': optimization_result['message'],
            'optimization_result': optimization_result
        }

    # Run optimization over free parameters
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_u,
        args=(unique_patterns, inference_method, param_names, fixed_settings),
        method=method,
        options={'disp': verbose, 'maxiter': 1000}
    )

    # Reconstruct fitted (transformed) parameters mixing optimized and fixed ones
    fitted_params = _params_from_reduced_unconstrained(result.x, param_names,fixed_settings)

    if verbose:
        print("\nOptimization complete!")
        print(f"Optimized unconstrained params (free only): {result.x}")
        print(f"Fitted params (transformed, including fixed): {fitted_params}")
        print(f"Final NLL: {result.fun:.4f}")
        print(f"Success: {result.success}, Message: {result.message}")

    return {
        'fitted_params': fitted_params,
        'nll': result.fun,
        'success': result.success,
        'message': result.message,
        'optimization_result': result
    }


base_params = ['s', 'temperature', 'p_keep']
structural_params = [
    ['br', 'cp_target', 'cp_source' ,'beta_A', 'beta_D'], # all same
    ['br_agent', 'br_non_agent', 'cp_agent_target', 'cp_non_agent_target', 'cp_agent_source', 'cp_non_agent_source', 'beta_A', 'beta_D'], # all different
    ['br', 'cp_agent_source', 'cp_non_agent_source','cp_agent_target' , 'cp_non_agent_target' ,'beta_A', 'beta_D'], # same base-rates but different powers
    ['br', 'cp_source','cp_agent_target' , 'cp_non_agent_target' ,'beta_A', 'beta_D'], # different target powers
    ['br', 'cp_agent_source', 'cp_non_agent_source','cp_target' ,'beta_A', 'beta_D'], # different source powers
    ['br_agent', 'br_non_agent', 'cp_source','cp_target' ,'beta_A', 'beta_D'], # different base-rates but same powers
    ['br_agent', 'br_non_agent', 'cp_target', 'cp_agent_source', 'cp_non_agent_source', 'beta_A', 'beta_D'], # same target power
    ['br_agent', 'br_non_agent', 'cp_agent_target', 'cp_non_agent_target', 'cp_source', 'beta_A', 'beta_D'], # same source power
]

structure_names = [
    'all_same',
    'all_different',
    'same_base_rates',
    'different_target_powers',
    'different_source_powers',
    'different_base_rates',
    'same_target_power',
    'same_source_power'
]

abalations = [
    {},# full
    {'p_keep':1}, # no commitment
    {'s':1}, # no backtracking
    {'s':1, 'p_keep':1} # no backtracking, no commitment
]

abalation_names = [
    'full',
    'p1',
    's1',
    's1_p1'
]

from CfModels.order_methods import static_twin

def static_twin_pow(**params):
    return static_twin(**params, smoothing='power')


import argparse
parser = argparse.ArgumentParser(description="Fit agency models.")
parser.add_argument("--use_power_method", type=int, default=0, choices=[0, 1], help="Use the power smoothing method for static_twin (0 or 1).")
parser.add_argument("--fix_beta_D", type=int, default=0, choices=[0, 1], help="Fix the target base-rate (beta_D) to 0 (0 or 1).")
args = parser.parse_args()

inference_method = static_twin_pow if args.use_power_method else static_twin

# Adjust structural params if fixing beta_D
if args.fix_beta_D:
    beta_D_setting = {'beta_D': 0}
else :
    beta_D_setting = {}

if args.use_power_method:
    inf_method = static_twin_pow
else:
    inf_method = static_twin


if args.fix_beta_D:
    beta_D_suffix = '_betaD0'
    print('Fixing beta_D to 0.')
else:
    print('Fitting beta_D freely.')
    beta_D_suffix = ''
if args.use_power_method:
    method_suffix = '_pow'
else:
    method_suffix = ''

import pickle

for i, structural_param_set in enumerate(structural_params):
    for j, ablation in enumerate(abalations):
        fit = fit_model(
            agency_data,
            inf_method,
            base_params + structural_param_set,
            fixed_settings=ablation | beta_D_setting,
            method = 'L-BFGS-B',
            verbose=1
        )
        
        with open(f'results/agency/agency_fit_{structure_names[i]}_{abalation_names[j]}{beta_D_suffix}{method_suffix}.pkl','wb') as f:
            pickle.dump(fit, f)


