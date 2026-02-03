import pandas as pd
import numpy as np

import os
import sys
# Ensure we can see the CfModels package (since notebook is in a subdir)
sys.path.append(os.path.abspath('..'))

import CfModels

# Define structures to fit

base_params = ['s', 'temperature', 'p_keep']

structural_params = [
    ['br'], # just one bease-rate for all nodes with causal powers 1-br
    ['br', 'causal_power'], # one base-rate and one causal power for all
    ['br', 'cp_source', 'cp_target'], # one base-rate different source and target probabilities 
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'cp_source', 'cp_target'], # different source and target probabilities 
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'causal_power'], # different base-rates one causal power
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'theta_AB', 'theta_AC', 'theta_BD', 'theta_CD'], # unconstrained
    ['br', 'beta_A', 'beta_D'], # br with fixed betas for B and C
    ['br', 'beta_A', 'beta_D', 'causal_power'], 
    ['br', 'beta_A', 'beta_D', 'cp_source', 'cp_target'], 
    ['br' ,'beta_A', 'beta_D', 'theta_AB', 'theta_AC', 'theta_BD', 'theta_CD'],
]

structure_names = [
    'br',
    'br_cp',
    'br_source_target',
    'betas_source_target',
    'betas_cp',
    'full',
    'BC',
    'BC_cp',
    'BC_source_target',
    'BC_full'
]

structures_dict = {name: params for name, params in zip(structure_names, structural_params)}

ablations = [
    {}, # full
    {'s':1}, # no backtracking
    {'p_keep':1}, # full commitment, backtracking
    {'p_keep':0}, # no commitment, backtracking
    {'p_keep':1, 's':1}, # full commitment, no backtracking
    {'p_keep':0, 's':1}, #  no commitment, no backtracking
]

ablation_names = [
    '',
    '_s1',
    '_p1',
    '_p0',
    '_s1_p1',
    '_s1_p0',
]

ablations_dict = {name: params for name, params in zip(ablation_names, ablations)}

# Load experiment 2 data
exp2_data = pd.read_csv('../../../data/data_2/experiment_2.csv')


# Define inference methods with different smoothing
def static_twin_pow(**params):
    return CfModels.order_methods.static_twin_with_missing(**params, smoothing='power')

def static_twin_exp(**params):
    return CfModels.order_methods.static_twin_with_missing(**params, smoothing='exponential')


# Check for any argument flags
import argparse
import pickle
parser = argparse.ArgumentParser(description='Cross-validate Experiment 2 models with different structures and ablations.')
parser.add_argument('--parallel', default=0, type=int, help='If set to 1, run fits in parallel using multiprocessing.')
parser.add_argument('--n_jobs', default=4, type=int, help='Number of parallel jobs to run if --parallel is set.')
parser.add_argument('--smoothing', default='exp', choices=['exp', 'pow'], help='Smoothing method for static_twin_with_missing.')
parser.add_argument('--optimizer', default='Powell', choices=['Powell', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP'], type=str, help='Optimizer to use for fitting.')
args = parser.parse_args()

if args.smoothing == 'exp':
    model_suffix = ''
    inf_method = static_twin_exp
    added_fixed = {}
else:
    # Case where we use the power function to get probability matching
    inf_method = static_twin_pow
    model_suffix = '_pm'
    added_fixed = {'temperature': 1}  # Example of adding a fixed param for power smoothing

data_dir = f'results/cross_validation/exp2_cond_' + args.optimizer.lower()
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Import joblib for parallel processing
from joblib import Parallel, delayed

# Define a function to perform cross-validation for a single model
def cross_validate_single_model(struct_name, struct_params, ablation_name, ablation_params, base_params, inf_method, added_fixed, data_dir, model_suffix, optimizer, exp2_data):
    model_name = f"{struct_name}{ablation_name}"
    
    # Check if results already exist
    results_file = f'{data_dir}/exp2_xval_{model_name}{model_suffix}.pkl'
    fit_file = f'exp2_fit_{model_name}{model_suffix}.pkl'
    if os.path.exists(results_file):
        print(f"Skipping {model_name} - cross-validation results already exist")
        return {'status': 'skipped', 'model': model_name}
    
    try:
        print(f"Cross-validating model: {model_name} with parameters: {base_params + struct_params} and ablation: {ablation_params}")
        
        # Set up cross-validation
        cv_result = CfModels.fitting.cross_validate_models(
            df=exp2_data,
            inference_method=inf_method,
            param_names=base_params + struct_params,
            fixed_settings={**ablation_params, **added_fixed},
            method=optimizer,
            full_fit_name=fit_file,
            verbose=1
        )
        
        # Save the cross-validation results
        with open(results_file, 'wb') as f:
            pickle.dump(cv_result, f)

        return {'status': 'completed', 'model': model_name, 'cv_result': cv_result}
    except Exception as e:
        print(f"Error in cross-validation for model {model_name}: {str(e)}")
        return {'status': 'error', 'model': model_name, 'error': str(e)}


# Run cross-validation either in parallel or sequentially
if args.parallel:
    print(f"Running cross-validation in parallel with {args.n_jobs} jobs...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(cross_validate_single_model)(
            struct_name,
            struct_params,
            ablation_name,
            ablation_params,
            base_params,
            inf_method,
            added_fixed,
            data_dir,
            model_suffix,
            args.optimizer,
            exp2_data
        )
        for struct_name, struct_params in structures_dict.items()
        for ablation_name, ablation_params in ablations_dict.items()
    )
else:
    print("Running cross-validation sequentially...")
    results = []
    
    # Cross-validate all combinations of structures and ablations
    for struct_name, struct_params in structures_dict.items():
        for ablation_name, ablation_params in ablations_dict.items():
            result = cross_validate_single_model(
                struct_name,
                struct_params,
                ablation_name,
                ablation_params,
                base_params,
                inf_method,
                added_fixed,
                data_dir,
                model_suffix,
                args.optimizer,
                exp2_data
            )
            results.append(result)

# Save summary of all results
summary_file = f'{data_dir}/xval_summary_{args.optimizer.lower()}_{model_suffix}.pkl'
with open(summary_file, 'wb') as f:
    pickle.dump(results, f)

print(f"Cross-validation complete. Results saved to {data_dir}")