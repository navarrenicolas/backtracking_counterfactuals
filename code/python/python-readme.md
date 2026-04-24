# Python README

## Folder structure

```
├── CfModels
│   ├── __init__.py
│   ├── fitting.py
│   ├── inference.py
│   ├── networks.py
│   ├── order_methods.py
│   ├── plotting.py
│   ├── utils.py
├── fitting
│   ├── fit_exp2_full.py
│   ├── xval_exp2_full.py
├── notebooks
│   ├── order_effects.ipynb
│   ├── analyses.ipynb
│   ├── results.ipynb
```

## Code Breakdown

### CfModels

This is a class for modeling counterfactual inference in probabilistic graphical models building on the ([pgmpy](https://pgmpy.org/)) library.
Extra modules for ease of use in other documents:
- Preset graphical representations: `CfModels/networks.py`
- Visualization tools: `CfModels/plotting.py`
- Extraneous functions: `CfModels/utils.py`

### notebooks

These are tutorials for using the `CfModels` module for sequential counterfactual inference.

#### `order_effects.ipynb`

This document provides an interactive demo of the order effect predictions while varying parameter values.
Further demonstration of the counterfactual inference and sequential inference process is described here.

#### `results.ipynb`

This document recreates the figure and tables from the paper using the model fit data in `../../data/model_fits/`

#### `analyses.ipynb`

This notebook contains supplementary analyses and model comparisons.

### fitting

These python scripts fit the counterfactual reasoning models to the order effects data found in 
`../data/data_2/experiment2.csv`. This is a cleaned file of the original data from Gerstenbeg et al.'s (2013) Experiment 2. The original data is in `../data/data_2/experiment2.xls`.

Both scripts make use of option flags to set the parameters of the fitting process. Below is a list of all the argument flags that can be chosen.


#### `fit_exp2_full.py` 

Flags:

```
--parallel (int, default: 0)
Set to 1 to run model fits in parallel across model configurations.

--n_jobs (int, default: 4)
Number of worker jobs used when --parallel is 1.

--smoothing (string, default: exp, choices: exp or pow)
Selects the smoothing rule in inference.
exp uses exponential smoothing.
pow uses power smoothing and fixes temperature to 1 internally.

--optimizer (string, default: Powell)
Optimization method used for fitting.
Choices: Powell, BFGS, L-BFGS-B, Nelder-Mead, CG, Newton-CG, TNC, COBYLA, SLSQP.
```

Example commands:

```
python fitting/fit_exp2_full.py
python fitting/fit_exp2_full.py --parallel 1 --n_jobs 8
python fitting/fit_exp2_full.py --smoothing pow
python fitting/fit_exp2_full.py --optimizer L-BFGS-B
```

#### `xval_exp2.py`
Flags:

```
--use_power_method (int, default: 0, choices: 0 or 1)
Set to 1 to use power smoothing instead of exponential smoothing.

--same_base_rates_BC (int, default: 0, choices: 0 or 1)
Set to 1 to use structures where B and C share the same base-rate setup.

--parallel (int, default: 0, choices: 0 or 1)
Set to 1 to parallelize cross-validation within each model fit.

--model_parallel (int, default: 0, choices: 0 or 1)
Set to 1 to parallelize across model configurations.

--n_model_jobs (int, default: 2)
Number of jobs used when --model_parallel is 1.

--n_cv_jobs (int, default: 2)
Number of jobs used for CV-level parallelization when --parallel is 1.

--cv_participants (int, default: 0, choices: 0 or 1)
Cross-validation scheme selector.
1 = participant-wise CV.
0 = condition-wise CV.

--no_commitment (int, default: 0, choices: 0 or 1)
Set to 1 to use no-commitment ablations (p_keep = 0 variants).

--use_paper_ablations (int, default: 0, choices: 0 or 1)
Set to 1 to use the ablation set matching the paper.
```

Example commands:
```
python fitting/xval_exp2.py
python fitting/xval_exp2.py --cv_participants 1
python fitting/xval_exp2.py --parallel 1 --n_cv_jobs 4
python fitting/xval_exp2.py --model_parallel 1 --n_model_jobs 6 --parallel 1 --n_cv_jobs 2
python fitting/xval_exp2.py --use_power_method 1 --use_paper_ablations 1
```




## References

Gerstenberg, T., Bechlivanidis, C., & Lagnado, D. A. (2013). Back on track: Backtracking in counterfactual reasoning. In Proceedings of the 35th Annual Conference of the Cognitive Science Society, Austin, TX, 2013 (pp. 2386-2391). Cognitive Science Society.

Ankur Ankan, & Johannes Textor (2024). pgmpy: A Python Toolkit for Bayesian Networks. Journal of Machine Learning Research, 25(265), 1-8.
