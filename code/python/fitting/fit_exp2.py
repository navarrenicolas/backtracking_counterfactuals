# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ...existing code...
import sys
import os

# Ensure we can see the CfModels package (since notebook is in a subdir)
sys.path.append(os.path.abspath('..'))


exp2_data = pd.read_csv('../../../data/data_2/experiment_2.csv')


from CfModels.order_methods import static_twin_with_missing
from CfModels.fitting import fit_model


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
    ['br' ,'beta_A', 'beta_D', 'theta_AB', 'theta_AC', 'theta_BD', 'theta_CD'],
]

structure_names_2 = [
    'BC',
    'BC_cp',
    'BC_source_target',
    'BC_full'
]

abalations = [{},# full
            {'p_keep':1}, # no commitment
            {'s':1}, # no backtracking
            {'s':1, 'p_keep':1} # no backtracking, no commitment
]

abalation_names = [
    '',
    'p1',
    's1',
    's1_p1'
]

import argparse
parser = argparse.ArgumentParser(description="Fit agency models.")
parser.add_argument("--use_power_method", type=int, default=0, choices=[0, 1], help="Use the power smoothing method for static_twin (0 or 1).")
parser.add_argument("--same_base_rates_BC", type=int, default=0, choices=[0, 1], help="Fit structures with the same base-rate for B and C (0 or 1).")
args = parser.parse_args()

if args.use_power_method:
    print('Using power method for smoothing')
    inf_method = static_twin_pow
    method_suffix = '_power'
else:
    print('Using exp method for smoothing')
    inf_method = static_twin_with_missing
    method_suffix = ''

if args.same_base_rates_BC:
    print('Using structures with same base rates for B and C')
    structural_params = structural_params_2
    structure_names = structure_names_2

import pickle

print('Starting fits...')

for i, structure in enumerate(structural_params):
    for j, ablation in enumerate(abalations):

        filename = f'exp2_fit_{structure_names[i]}_{abalation_names[j]}{method_suffix}.pkl'
        # check if file exists
        if os.path.exists(f'results/exp2/{filename}'):
            print(f"File {filename} already exists. Skipping fit.")
            continue

        print (f'Fitting structure {structure_names[i]} with ablation {abalation_names[j]}')
        fit = fit_model(
            exp2_data,
            inf_method,
            base_params + structure,
            fixed_settings=ablation,
            method = 'Powell',
            verbose=1
        )
        with open(f'results/exp2/{filename}','wb') as f:
            pickle.dump(fit, f)
