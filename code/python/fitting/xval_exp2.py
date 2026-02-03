import numpy as np
import pandas as pd

# ...existing code...
import sys
import os
import pickle

sys.path.append(os.path.abspath('..'))

from CfModels.order_methods import static_twin_with_missing
from CfModels.fitting import cross_validate_participants, cross_validate_participants_parallel, cross_validate_models_parallel, cross_validate_models

from joblib import Parallel, delayed


exp2_data = pd.read_csv('../../../data/data_2/experiment_2.csv')

def static_twin_pow(**params):
    return static_twin_with_missing(**params, smoothing='power') 


base_params = ['s', 'temperature', 'p_keep']

structural_params = [
    ['br'], # just one bease-rate for all nodes with causal powers 1-br
    ['br', 'causal_power'], # one base-rate and one causal power for all nodes
    ['br', 'cp_source', 'cp_target'], # one base-rate different source and target probabilities 
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'cp_source', 'cp_target'], # different source and target probabilities 
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'causal_power'], # different base-rates one causal power
    ['beta_B', 'beta_C' ,'beta_A', 'beta_D', 'theta_AB', 'theta_AC', 'theta_BD', 'theta_CD'], # unconstrained
]

structure_names = [
    'br',
    'br_cp',
    'br_source_target',
    'betas_source_target',
    'betas_cp',
    'full'
]

structural_params_2 = [
    ['br', 'beta_A', 'beta_D'], # br with fixed betas for B and C
    ['br', 'beta_A', 'beta_D', 'causal_power'], 
    ['br', 'beta_A', 'beta_D', 'cp_source', 'cp_target'], 
    ['br','beta_A', 'beta_D', 'theta_AB', 'theta_AC', 'theta_BD', 'theta_CD'],
]

structure_names_2 = [
    'BC',
    'BC_cp',
    'BC_source_target',
    'BC_full'
]

ablations = [{},# full
            {'p_keep':1}, # no commitment
            {'s':1}, # no backtracking
            {'s':1, 'p_keep':1} # no backtracking, no commitment
]

ablation_names = [
    '',
    'p1',
    's1',
    's1_p1'
]

ablations_2 = [
    {'p_keep':0}, # no commitment, fixed betas for B and C
    {'s':1, 'p_keep':0}, # no backtracking, no commitment
]

ablation_names_2 = [
    'p0',
    's1_p0'
]

ablations_paper = [
    {'p_keep':0, 's':1}, # no commitment, no backtracking
    {'p_keep':0}, # no commitment, backtracking
    {'p_keep':1, 's':1}, # commitment, no backtracking
    {'p_keep':1}, # commitment, backtracking
]

ablation_names_paper = [
    's1_p0',
    'p0',
    's1_p1',
    'p1'
]

import argparse
parser = argparse.ArgumentParser(description="Fit agency models.")
parser.add_argument("--use_power_method", type=int, default=0, choices=[0, 1], help="Use the power smoothing method for static_twin (0 or 1).")
parser.add_argument("--same_base_rates_BC", type=int, default=0, choices=[0, 1], help="Fit structures with the same base-rate for B and C (0 or 1).")
parser.add_argument("--parallel", type=int, default=0, choices=[0, 1], help="Run cross-validation in parallel (0 or 1).")
parser.add_argument("--model_parallel", type=int, default=0, choices=[0, 1], help="Run model fitting in parallel (0 or 1).")
parser.add_argument("--n_model_jobs", type=int, default=2, help="Number of parallel model jobs (default: 2).")
parser.add_argument("--n_cv_jobs", type=int, default=2, help="Number of parallel CV jobs per model (default: 2).")
parser.add_argument("--cv_participants", type=int, default=0, choices=[0, 1], help="Pick cross validation scheme (0 or 1).")
parser.add_argument('--no_commitment', type=int, default=0, choices=[0, 1], help='Ablate commitment by setting p_keep=0.')
parser.add_argument('--use_paper_ablations', type=int, default=0, choices=[0, 1], help='Use ablations from the paper (0 or 1).')
args = parser.parse_args()

# Determine cross-validation method and directory
if args.cv_participants:
    print('Using participant-wise cross-validation')
    parallel_cv = cross_validate_participants_parallel
    serial_cv = cross_validate_participants
    cv_dir = 'exp2'
else:
    print('Using condition-wise cross-validation')
    parallel_cv = cross_validate_models_parallel
    serial_cv = cross_validate_models
    cv_dir = 'exp2_cond'

# Set model fitting method based on decision rule
if args.use_power_method:
    print('Using power method for smoothing')
    inf_method = static_twin_pow
    method_suffix = '_power'
else:
    print('Using exp method for smoothing')
    inf_method = static_twin_with_missing
    method_suffix = ''

# Adjust structures and names if using same base rates for B and C
if args.same_base_rates_BC:
    print('Using structures with same base rates for B and C')
    structural_params = structural_params_2
    structure_names = structure_names_2

# Adjust ablations if no commitment is specified or if using paper ablations
if args.use_paper_ablations:
    print('Using ablations from the paper')
    ablations = ablations_paper
    ablation_names = ablation_names_paper
elif args.no_commitment:
    print('Ablating commitment by setting p_keep=0')
    ablations = ablations_2
    ablation_names = ablation_names_2


def run_single_model_cv(model_config):
    """
    Run cross-validation for a single model configuration.
    This function needs to be at module level for joblib pickling.
    """
    (exp2_data, inf_method, base_params, structure, structure_name, 
     ablation, ablation_name, method_suffix, use_parallel_cv, method) = model_config
    
    filename = f'exp2_fit_{structure_name}_{ablation_name}{method_suffix}.pkl'
    xval_name = f'exp2_xval_{structure_name}_{ablation_name}{method_suffix}.pkl'
    

    print(f'Fitting structure {structure_name} with ablation {ablation_name}')
    
    try:
        if use_parallel_cv:
            cv_result = parallel_cv(
                exp2_data,
                inf_method,
                base_params + structure,
                fixed_settings=ablation,
                full_fit_name=filename,
                method=method,
                n_jobs=args.n_cv_jobs,  # Use the number of CV jobs from arguments
                verbose=0  # Reduced verbosity for cleaner parallel output
            )
        else:
            cv_result = serial_cv(
                exp2_data,
                inf_method,
                base_params + structure,
                fixed_settings=ablation,
                full_fit_name=filename,
                method=method,
                verbose=0  # Reduced verbosity for cleaner parallel output
            )
        
        # Save the result
    
        os.makedirs(f'results/cross_validation/{cv_dir}/', exist_ok=True)
        with open(f'results/cross_validation/{cv_dir}/{xval_name}', 'wb') as f:
            pickle.dump(cv_result, f)
        
        return {
            'structure_name': structure_name,
            'ablation_name': ablation_name,
            'success': True,
            'error': None,
            'cv_result_length': len(cv_result) if cv_result else 0
        }
        
    except Exception as e:
        print(f'Error fitting {structure_name}_{ablation_name}: {str(e)}')
        return {
            'structure_name': structure_name,
            'ablation_name': ablation_name,
            'success': False,
            'error': str(e),
            'cv_result_length': 0
        }


print('Starting cross validation fits...')

if args.model_parallel:
    print(f'Running model fitting in parallel with {args.n_model_jobs} jobs')
    print(f'Individual CV parallelization: {"ON" if args.parallel else "OFF"}')
    
    # Prepare all model configurations
    model_configs = []
    for i, structure in enumerate(structural_params):
        for j, ablation in enumerate(ablations):

            # Check if result file already exists
            xval_name = f'exp2_xval_{structure_names[i]}_{ablation_names[j]}{method_suffix}.pkl'
            result_path = f'results/cross_validation/{cv_dir}/{xval_name}'
            
            if os.path.exists(result_path):
                print(f'Skipping {structure_names[i]}_{ablation_names[j]} - file exists: {result_path}')
                continue

            model_configs.append((
                exp2_data, inf_method, base_params, structure, structure_names[i],
                ablation, ablation_names[j], method_suffix, args.parallel, 'Powell'
            ))
    
    print(f'Total model configurations to fit: {len(model_configs)}')
    
    # Run model fits in parallel
    results = Parallel(n_jobs=args.n_model_jobs, verbose=1)(
        delayed(run_single_model_cv)(config) for config in model_configs
    )
    
    # Report results
    successful_fits = sum(1 for r in results if r['success'])
    print(f'\n=== PARALLEL MODEL FITTING COMPLETE ===')
    print(f'Successfully completed: {successful_fits}/{len(results)} model configurations')
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        structure_name = result['structure_name']
        ablation_name = result['ablation_name']
        if result['success']:
            print(f'{status} {structure_name}_{ablation_name} ({result["cv_result_length"]} CV folds)')
        else:
            print(f'{status} {structure_name}_{ablation_name} - ERROR: {result["error"]}')

else:
    # Original sequential approach
    if args.parallel:
        print('Running cross-validation in parallel (sequential models)')
        for i, structure in enumerate(structural_params):
            for j, ablation in enumerate(ablations):

                # Check if result file already exists
                xval_name = f'exp2_xval_{structure_names[i]}_{ablation_names[j]}{method_suffix}.pkl'
                result_path = f'results/cross_validation/{cv_dir}/{xval_name}'
                if os.path.exists(result_path):
                    print(f'Skipping {structure_names[i]}_{ablation_names[j]} - file exists: {result_path}')
                    continue

                filename = f'exp2_fit_{structure_names[i]}_{ablation_names[j]}{method_suffix}.pkl'


                print (f'Fitting structure {structure_names[i]} with ablation {ablation_names[j]} in parallel')
                fit = parallel_cv(
                    exp2_data,
                    inf_method,
                    base_params + structure,
                    fixed_settings=ablation,
                    full_fit_name=filename,
                    method = 'Powell',
                    n_jobs=args.n_cv_jobs,
                    verbose=1
                )
                os.makedirs(f'results/cross_validation/{cv_dir}/', exist_ok=True)
                with open(f'results/cross_validation/{cv_dir}/{xval_name}','wb') as f:
                    pickle.dump(fit, f)
    else:
        print('Running cross-validation sequentially (no parallelization)')
        for i, structure in enumerate(structural_params):
            for j, ablation in enumerate(ablations):

                # Check if result file already exists
                xval_name = f'exp2_xval_{structure_names[i]}_{ablation_names[j]}{method_suffix}.pkl'
                result_path = f'results/cross_validation/{cv_dir}/{xval_name}'
                if os.path.exists(result_path):
                    print(f'Skipping {structure_names[i]}_{ablation_names[j]} - file exists: {result_path}')
                    continue
                filename = f'exp2_fit_{structure_names[i]}_{ablation_names[j]}{method_suffix}.pkl'

                print (f'Fitting structure {structure_names[i]} with ablation {ablation_names[j]}')
                fit = serial_cv(
                    exp2_data,
                    inf_method,
                    base_params + structure,
                    fixed_settings=ablation,
                    full_fit_name=filename,
                    method = 'Powell',
                    verbose=1
                )
                os.makedirs(f'results/cross_validation/{cv_dir}/', exist_ok=True)
                with open(f'results/cross_validation/{cv_dir}/{xval_name}','wb') as f:
                    pickle.dump(fit, f)
