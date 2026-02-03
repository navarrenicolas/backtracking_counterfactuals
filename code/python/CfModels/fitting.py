from hashlib import new
import numpy as np

from scipy.optimize import minimize

from .utils import *
from .inference import CounterfactualInferenceBN
from .networks import create_diamond_bayesian_network
import os
import pickle

from joblib import Parallel, delayed

def convert_data(row):
    response_map = {0:None,1:1,-1:0}
    A_resp = response_map[row.A]
    C_resp = response_map[row.C]
    D_resp = response_map[row.D]
    if row.order == 1:
        path = [{'A': A_resp}, {'C': C_resp}, {'D': D_resp}]
    else:
        path = [{'D': D_resp}, {'C': C_resp}, {'A': A_resp}]
    return path

def extract_for_fit(row):
    path = convert_data(row)
    disjunction = row.structure == 1
    question = row.question == 1
    fit_dict = {
        'path': path,
        'disjunction': disjunction,
        'question': question
    }
    return fit_dict


def unique_pattern_analysis(df,verbose = False):
    
    print('Extracting unique patterns from data...')
    # Generate the list of extracted data
    data_list = [extract_for_fit(row) for _, row in df.iterrows()]

    # Create a manual counter using a dictionary
    pattern_counts = {}

    for item in data_list:
        # Convert the path list and other elements to a string representation for comparison
        path_str = str([(list(d.keys())[0], d[list(d.keys())[0]]) for d in item['path']])
        pattern_key = (path_str, item['disjunction'], item['question'])
        
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
    eps = 1e-6 # to avoid extreme values of 0 or 1
    sigmoid_max = 1 - eps
    new_range = sigmoid_max - eps

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
                # rescale sigmoid from (0,1) to (eps, 1-eps)
                out[name] = eps + sigmoid(x) * new_range
    return out


def make_priors(transformed):
    """
    Extract full set of model parameters from transformed dict.
    """
    
    br = transformed.get('br',None)
    cp  = transformed.get('causal_power', None)
    cp_source = transformed.get('cp_source', None)
    cp_target = transformed.get('cp_target', None)
    beta_A = transformed.get('beta_A', None)
    beta_C = transformed.get('beta_C', None)
    beta_B = transformed.get('beta_B', None)
    beta_D = transformed.get('beta_D', None)
    theta_AB = transformed.get('theta_AB', None)
    theta_AC = transformed.get('theta_AC', None)
    theta_BD = transformed.get('theta_BD', None)
    theta_CD = transformed.get('theta_CD', None)
        

    if beta_B is None:
        beta_B = br
    if beta_C is None:
        beta_C = br
    if beta_A is None:
        beta_A = br
    if beta_D is None:
        beta_D = br
    
    if cp is None:
        if cp_source is None:
            theta_AB = 1-br if theta_AB is None else theta_AB
            theta_AC = 1-br if theta_AC is None else theta_AC
        else:
            theta_AB = cp_source 
            theta_AC = cp_source 

        if cp_target is None :
            theta_BD = 1-br if theta_BD is None else theta_BD
            theta_CD = 1-br if theta_CD is None else theta_CD
        else:
            theta_BD = cp_target 
            theta_CD = cp_target
    else:
        if cp_source is None:
            theta_AB = cp if theta_AB is None else theta_AB
            theta_AC = cp if theta_AC is None else theta_AC
        else: 
            theta_AB = cp_source
            theta_AC = cp_source
        if cp_target is None :
            theta_BD = cp if theta_BD is None else theta_BD
            theta_CD = cp if theta_CD is None else theta_CD
        else:
            theta_BD = cp_target
            theta_CD = cp_target


    # Validate that all required parameters are set
    return {
        'beta_A': beta_A,
        'beta_B': beta_B,
        'beta_C': beta_C,
        'beta_D': beta_D,
        'theta_AB': theta_AB,
        'theta_AC': theta_AC,
        'theta_BD': theta_BD,
        'theta_CD': theta_CD
    }


# Add a simple cache for inference engines
_inference_cache = {}

def _get_inference_engines(theta_AB, theta_AC, theta_BD, theta_CD, 
                           beta_A, beta_B, beta_C, beta_D):
    """Cache and reuse inference engines for identical parameter configurations."""
    cache_key = (theta_AB, theta_AC, theta_BD, theta_CD, beta_A, beta_B, beta_C, beta_D)
    
    if cache_key not in _inference_cache:
        disj_model = create_diamond_bayesian_network(
            power_B=theta_AB, power_C=theta_AC,
            power_D_from_B=theta_BD, power_D_from_C=theta_CD,
            beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
            disjunction=True
        )
        conj_model = create_diamond_bayesian_network(
            power_B=theta_AB, power_C=theta_AC,
            power_D_from_B=theta_BD, power_D_from_C=theta_CD,
            beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
            disjunction=False
        )
        _inference_cache[cache_key] = {
            'disj_inference': CounterfactualInferenceBN(disj_model),
            'conj_inference': CounterfactualInferenceBN(conj_model)
        }

    return _inference_cache[cache_key]


def negative_log_likelihood(params, unique_patterns, inference_method, param_names, fixed_settings={}, random_pred = False, verbose=False):
    """
    Now 'params' are unconstrained optimizer variables (real numbers) for the free parameters.
    They are transformed to the model domain here. fixed_settings can contain any subset of
    the model parameters (s, p_keep, br, beta_A, temperature) to hold fixed.
    """
    try:
        transformed = _params_from_reduced_unconstrained(params, param_names, fixed_settings)
    except Exception as e:
        if verbose:
            print("Parameter reconstruction error:", e)
        return np.inf


    s = transformed['s']
    p_keep = transformed['p_keep']
    temperature = transformed['temperature']


        
    required_params = make_priors(transformed)
    
    none_params = [name for name, value in required_params.items() if value is None]
    if none_params:
        raise ValueError(f"The following required parameters are None: {', '.join(none_params)}")

    # Build models with current parameters (command/outcome powers and base rates may be in fixed_settings)
    inference_engines = _get_inference_engines(
        theta_AB = required_params['theta_AB'],
        theta_AC=required_params['theta_AC'],
        theta_BD=required_params['theta_BD'],
        theta_CD=required_params['theta_CD'],
        beta_A=required_params['beta_A'],
        beta_B=required_params['beta_B'],
        beta_C=required_params['beta_C'],
        beta_D=required_params['beta_D']
    )
     

    # Compute log-likelihood (sum over pattern counts)
    ll = 0.0

    for pattern, count in unique_patterns:

        if random_pred:
            # for random prediction, we ignore model and return uniform probability
            prob = 1.0/8
            ll += np.log(prob) * count
            continue

        path = [dict([i]) for i in eval(pattern[0])]
        disjunction = pattern[1]
        question = pattern[2]
        cf_constraint = {'B':1} if question else {'B':0}
        obs = {'A':0,'B':0,'C':0,'D':0} if question else {'A':1,'B':1,'C':1,'D':1}
        
        result = inference_method(
            base_inf= inference_engines['disj_inference'] if disjunction else inference_engines['conj_inference'],
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

def nll_quick(params, df, inference_method, random_pred = False, verbose=False):
    """
    Wrapper to compute NLL directly from dataframe.
    """
    unique_patterns = unique_pattern_analysis(df,verbose=verbose)
    
    s = params['s']
    p_keep = params['p_keep']
    temperature = params['temperature']

    required_params = make_priors(params)
    
    none_params = [name for name, value in required_params.items() if value is None]
    if none_params:
        raise ValueError(f"The following required parameters are None: {', '.join(none_params)}")

    # Build models with current parameters (command/outcome powers and base rates may be in fixed_settings)
    inference_engines = _get_inference_engines(
        theta_AB=required_params['theta_AB'], 
        theta_AC=required_params['theta_AC'], 
        theta_BD=required_params['theta_BD'], 
        theta_CD=required_params['theta_CD'],
        beta_A=required_params['beta_A'],
        beta_B=required_params['beta_B'],
        beta_C=required_params['beta_C'], 
        beta_D=required_params['beta_D']
    )
     

    # Compute log-likelihood (sum over pattern counts)
    ll = 0.0

    for pattern, count in unique_patterns:

        if random_pred:
            # for random prediction, we ignore model and return uniform probability
            prob = 1.0/8
            ll += np.log(prob) * count
            continue

        path = [dict([i]) for i in eval(pattern[0])]
        disjunction = pattern[1]
        question = pattern[2]
        cf_constraint = {'B':1} if question else {'B':0}
        obs = {'A':0,'B':0,'C':0,'D':0} if question else {'A':1,'B':1,'C':1,'D':1}
        
        result = inference_method(
            base_inf= inference_engines['disj_inference'] if disjunction else inference_engines['conj_inference'],
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


def fit_model(df, inference_method, param_names=None, fixed_settings={}, method='Powell', fit_file_name = None, verbose=True):
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


###################
## CROSS VALIDATION
###################

def cross_validate_participants(df, inference_method, param_names=None, fixed_settings={}, method='Powell', full_fit_name = None, verbose=True):
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

    if full_fit_name:
        warm_start_path = os.path.join('results', 'exp2', full_fit_name)
        if os.path.exists(warm_start_path):
            if verbose:
                print(f"Loading precomputed fit from {warm_start_path} for warm start...")
            try:
                with open(warm_start_path, 'rb') as f:
                    fit_data = pickle.load(f)
                
                # Extract unconstrained parameters (x) from the optimization result
                if 'optimization_result' in fit_data:
                    loaded_u = fit_data['optimization_result'].x
                    if len(loaded_u) == n_free:
                        warm_start_u = loaded_u
                        # If we have a warm start, we update initial_u. 
                        # The code below will use this initial_u as x0 for the full fit check.
                    else:
                        if verbose:
                            print(f"Warm start parameter count mismatch. Expected {n_free}, got {len(loaded_u)}.")
            except Exception as e:
                if verbose:
                    print(f"Error loading warm start file: {e}")
    
    # Optional warm start from full dataset fit
    if n_free > 0 and 'warm_start_u' not in locals():
        warm_start_u = initial_u
    
        if verbose:
            print("Running full dataset fit to initialize cross-validation folds (Warm Start)...")
        full_res = minimize(
            fun=negative_log_likelihood,
            x0=warm_start_u,
            args=(unique_patterns, inference_method, param_names, fixed_settings),
            method=method,
            options={'disp': verbose, 'maxiter': 1000}
        )
        warm_start_u = full_res.x

        fitted_params = _params_from_reduced_unconstrained(warm_start_u, param_names,fixed_settings)


        if verbose:
            print("Warm start optimization complete.")
            print(f"Warm start unconstrained params (free only): {warm_start_u}")
            print(f"Warm start NLL: {full_res.fun:.4f}")
    
        # Save warm start to results/exp2/
        try:
            warm_start_path = os.path.join('results', 'exp2', full_fit_name)
            # Ensure directory exists
            os.makedirs(os.path.dirname(warm_start_path), exist_ok=True)
            
            with open(warm_start_path, 'wb') as f:
                pickle.dump(         
                    {'fitted_params': fitted_params,
                    'nll': full_res.fun,
                    'success': full_res.success,
                    'message': full_res.message,
                    'optimization_result': full_res
                    }, f)
        except Exception as e:
            print(f"Error saving warm start file: {e}")

    
    if verbose:
        print("Starting leave-one-pattern-out cross-validation...")
    
    results = []


    for idx,(p,count) in enumerate(unique_patterns):
    
        removed_patterns = unique_patterns.copy()
        removed_patterns[idx] = (p,count-1)
    
        # Run optimization over free parameters
        result = minimize(
            fun=negative_log_likelihood,
            x0=warm_start_u,
            args=(removed_patterns, inference_method, param_names, fixed_settings),
            method=method,
            options={'disp': False, 'maxiter': 1000}
        )

        # Reconstruct fitted (transformed) parameters mixing optimized and fixed ones
        fitted_params = _params_from_reduced_unconstrained(result.x, param_names,fixed_settings)

        cv_result = negative_log_likelihood(
            params=result.x,
            unique_patterns=[(p,1)],
            inference_method=inference_method,
            param_names=param_names,
            fixed_settings=fixed_settings,
            verbose=False
        )

        results.append({
            'fitted_params': fitted_params,
            'nll': cv_result,
            'success': result.success,
            'message': result.message,
            'pattern': p,
            'count': count,
            'optimization_result': result
        })
    
    if verbose:
        print("\nCross-validation complete!")

    return results        




def _fit_single_cv_fold(idx, pattern_count_tuple, removed_patterns, warm_start_u, 
                       inference_method, param_names, fixed_settings, method,
                       verbose):
    """
    Fit a single cross-validation fold. This function needs to be at module level
    for proper pickling by joblib.
    """
    p, count = pattern_count_tuple
    
    # Run optimization over free parameters
    result = minimize(
        fun=negative_log_likelihood,
        x0=warm_start_u,
        args=(removed_patterns, inference_method, param_names, fixed_settings),
        method=method,
        options={'disp': False, 'maxiter': 1000}
    )

    # Reconstruct fitted (transformed) parameters
    fitted_params = _params_from_reduced_unconstrained(result.x, param_names, fixed_settings)

    # Evaluate on held-out pattern
    cv_result = negative_log_likelihood(
        params=result.x,
        unique_patterns=[(p, 1)],
        inference_method=inference_method,
        param_names=param_names,
        fixed_settings=fixed_settings,
        verbose=False
    )

    if verbose:
        print(f"CV Fold {idx+1}: Pattern {p}, Count {count}, NLL {cv_result:.4f}, Success: {result.success}")

    return {
        'fitted_params': fitted_params,
        'nll': cv_result,
        'success': result.success,
        'message': result.message,
        'pattern': p,
        'count': count,
        'optimization_result': result,
        'fold_idx': idx
    }

def cross_validate_participants_parallel(df, inference_method, param_names=None, 
                                       fixed_settings={}, method='Powell', 
                                       full_fit_name=None, n_jobs=-1, verbose=True):
    """
    Parallel version of cross_validate_participants using joblib.
    
    Args:
        n_jobs (int): Number of parallel jobs. -1 uses all available cores.
                     Use 1 for serial execution, 2-4 for moderate parallelism.
    """
    # ... existing setup code (same as original) ...
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

    # Warm start logic (same as original)
    if full_fit_name:
        warm_start_path = os.path.join('results', 'exp2', full_fit_name)
        if os.path.exists(warm_start_path):
            if verbose:
                print(f"Loading precomputed fit from {warm_start_path} for warm start...")
            try:
                with open(warm_start_path, 'rb') as f:
                    fit_data = pickle.load(f)
                
                if 'optimization_result' in fit_data:
                    loaded_u = fit_data['optimization_result'].x
                    if len(loaded_u) == n_free:
                        warm_start_u = loaded_u
                    else:
                        if verbose:
                            print(f"Warm start parameter count mismatch. Expected {n_free}, got {len(loaded_u)}.")
            except Exception as e:
                if verbose:
                    print(f"Error loading warm start file: {e}")
    
    # Warm start computation (same as original)
    if n_free > 0 and 'warm_start_u' not in locals():
        warm_start_u = initial_u
    
        if verbose:
            print("Running full dataset fit to initialize cross-validation folds (Warm Start)...")
        full_res = minimize(
            fun=negative_log_likelihood,
            x0=warm_start_u,
            args=(unique_patterns, inference_method, param_names, fixed_settings),
            method=method,
            options={'disp': verbose, 'maxiter': 1000}
        )
        warm_start_u = full_res.x
        fitted_params = _params_from_reduced_unconstrained(warm_start_u, param_names, fixed_settings)

        if verbose:
            print("Warm start optimization complete.")
            print(f"Warm start unconstrained params (free only): {warm_start_u}")
            print(f"Warm start NLL: {full_res.fun:.4f}")
    
        # Save warm start (same as original)
        try:
            warm_start_path = os.path.join('results', 'exp2', full_fit_name)
            os.makedirs(os.path.dirname(warm_start_path), exist_ok=True)
            
            with open(warm_start_path, 'wb') as f:
                pickle.dump({
                    'fitted_params': fitted_params,
                    'nll': full_res.fun,
                    'success': full_res.success,
                    'message': full_res.message,
                    'optimization_result': full_res
                }, f)
        except Exception as e:
            print(f"Error saving warm start file: {e}")

    if verbose:
        print(f"Starting parallel leave-one-pattern-out cross-validation with {n_jobs} jobs...")
    
    # Prepare data for parallel processing
    cv_tasks = []
    for idx, (p, count) in enumerate(unique_patterns):
        removed_patterns = unique_patterns.copy()
        removed_patterns[idx] = (p, count-1)
        cv_tasks.append((idx, (p, count), removed_patterns))
    
    # Run parallel cross-validation
    results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(_fit_single_cv_fold)(
            idx, pattern_count_tuple, removed_patterns, warm_start_u,
            inference_method, param_names, fixed_settings, method, verbose
        )
        for idx, pattern_count_tuple, removed_patterns in cv_tasks
    )
    
    # Sort results by fold index to maintain order
    results = sorted(results, key=lambda x: x['fold_idx'])
    
    # Remove fold_idx as it was only needed for sorting
    for result in results:
        del result['fold_idx']
    
    if verbose:
        print("\nParallel cross-validation complete!")
        successful_fits = sum(1 for r in results if r['success'])
        print(f"Successful fits: {successful_fits}/{len(results)}")

    return results

def cross_validate_models(df, inference_method, param_names=None, fixed_settings={}, method='Powell', full_fit_name=None, verbose=True):
    """
    Perform leave-one-condition-out cross-validation for counterfactual inference models.
    
    There are 8 conditions of interest:
    - order (ACD=1/DCA=2) × question (off=0/on=1) × structure (disjunctive=1/conjunctive=2)
    
    Args:
        df (pd.DataFrame): Full dataset with columns ['order', 'question', 'structure', 'A', 'C', 'D']
        inference_method (callable): Inference function
        param_names (list): Parameter names to fit (default: ['s', 'p_keep', 'br', 'beta_A', 'temperature'])
        fixed_settings (dict): Parameters to fix during fitting
        method (str): Optimization method for scipy.minimize
        full_fit_name (str): Filename to save/load full dataset fit for warm starting.
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Cross-validation results with train/test NLL for each condition
    """
    if param_names is None:
        param_names = ['s', 'p_keep', 'br', 'beta_A', 'temperature']

    # Determine free parameters
    free_param_names = [n for n in param_names if n not in fixed_settings]
    n_free = len(free_param_names)

    # Initialize warm start
    warm_start_u = None
    if n_free > 0:
        initial_u = np.random.randn(n_free)
    else:
        initial_u = np.array([])
    warm_start_u = initial_u

    # Warm start logic (load existing fit)
    if full_fit_name:
        warm_start_path = os.path.join('results', 'exp2', full_fit_name)
        if os.path.exists(warm_start_path):
            if verbose:
                print(f"Loading precomputed fit from {warm_start_path} for warm start...")
            try:
                with open(warm_start_path, 'rb') as f:
                    fit_data = pickle.load(f)
                
                if 'optimization_result' in fit_data and hasattr(fit_data['optimization_result'], 'x'):
                    loaded_u = fit_data['optimization_result'].x # or ['x'] if dict
                    # Handling both dict and object
                    if isinstance(fit_data['optimization_result'], dict):
                         loaded_u = fit_data['optimization_result']['x']
                    
                    if len(loaded_u) == n_free:
                        warm_start_u = loaded_u
                    else:
                        if verbose:
                            print(f"Warm start parameter count mismatch. Expected {n_free}, got {len(loaded_u)}.")
            except Exception as e:
                if verbose:
                    print(f"Error loading warm start file: {e}")

    # If warm start not loaded but requested (or just to have a good starting point), run full fit
    # Note: For CV models, running a full fit first is beneficial as a warm start for folds
    # We only run full fit if we didn't load a valid warm_start_u or if we want to ensure we have one.
    # To mimic participants logic: if full fit is not loaded, we compute it.
    
    # Check if we actually loaded something different from initial random
    is_loaded = not np.array_equal(warm_start_u, initial_u) if n_free > 0 else True

    if full_fit_name and not is_loaded and n_free > 0:
        if verbose:
            print("Running full dataset fit to initialize cross-validation folds (Warm Start)...")
        
        full_fit_result = fit_model(
            df=df,
            inference_method=inference_method,
            param_names=param_names,
            fixed_settings=fixed_settings,
            method=method,
            verbose=verbose
        )
        
        if full_fit_result['success'] or full_fit_result['optimization_result'].success:
             warm_start_u = full_fit_result['optimization_result'].x
        
        # Save warm start
        try:
            warm_start_path = os.path.join('results', 'exp2', full_fit_name)
            os.makedirs(os.path.dirname(warm_start_path), exist_ok=True)
            with open(warm_start_path, 'wb') as f:
                pickle.dump(full_fit_result, f)
            if verbose:
                print(f"Saved warm start fit to {warm_start_path}")
        except Exception as e:
            print(f"Error saving warm start file: {e}")

    # Define all 8 conditions
    conditions = []
    for order in [1, 2]:  # ACD, DCA
        for question in [0, 1]:  # off, on
            for structure in [1, 2]:  # disjunctive, conjunctive
                conditions.append((order, question, structure))
    
    if verbose:
        print(f"Cross-validation with {len(conditions)} conditions")
        print("-" * 60)
    
    # Store results for each method
    method_results = {
        'conditions': [],
        'train_nlls': [],
        'test_nlls': [],
        'fitted_params': [],
        'success_rates': []
    }
    
    for i, test_condition in enumerate(conditions):
        order_test, question_test, structure_test = test_condition
        
        # Create train/test splits
        test_mask = ((df['order'] == order_test) & 
                    (df['question'] == question_test) & 
                    (df['structure'] == structure_test))
        
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        
        if verbose:
            condition_name = f"Order={order_test}, Question={question_test}, Structure={structure_test}"
            print(f"\nFold {i+1}/{len(conditions)}: {condition_name}")
            print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Fit model on training data
        try:
            if verbose:
                print("  Fitting model on training data...")
            
            # Since fit_model generates random start internally, we need to pass x0 if we want warm start.
            # However, fit_model currently initializes random x0. 
            # To support warm start in fit_model, we would need to change fit_model or minimize directly here.
            # To avoid changing fit_model signature elsewhere, let's replicate the fit call here using minimize directly
            # similar to cross_validate_participants.
            
            unique_patterns_train = unique_pattern_analysis(train_df, verbose=False)
            
            # Use warm_start_u as initial guess
            res = minimize(
                fun=negative_log_likelihood,
                x0=warm_start_u,
                args=(unique_patterns_train, inference_method, param_names, fixed_settings),
                method=method,
                options={'disp': False, 'maxiter': 1000}
            )

            train_nll = res.fun
            fitted_params = _params_from_reduced_unconstrained(res.x, param_names, fixed_settings)
            fit_success = res.success
            optimized_u = res.x

            # Calculate Test NLL
            unique_patterns_test = unique_pattern_analysis(test_df, verbose=False)
            
            test_nll = negative_log_likelihood(
                params=optimized_u,
                unique_patterns=unique_patterns_test,
                inference_method=inference_method,
                param_names=param_names,
                fixed_settings=fixed_settings,
                verbose=False
            )
            
            if verbose:
                print(f"  Train NLL: {train_nll:.4f}, Test NLL: {test_nll:.4f}, Success: {fit_success}")
            
                    
        except Exception as e:
            if verbose:
                print(f"  Training failed: {e}")
            train_nll = np.inf
            test_nll = np.inf
            fitted_params = None
            fit_success = False
        
        # Store results
        method_results['conditions'].append(test_condition)
        method_results['train_nlls'].append(train_nll)
        method_results['test_nlls'].append(test_nll)
        method_results['fitted_params'].append(fitted_params)
        method_results['success_rates'].append(fit_success)
        
    return method_results

def _fit_single_condition_fold(i, test_condition, train_df, test_df, warm_start_u, inference_method, 
                                param_names, fixed_settings, method, verbose):
    """
    Fit a single condition fold.
    """
    order_test, question_test, structure_test = test_condition
    
    if verbose:
        condition_name = f"Order={order_test}, Question={question_test}, Structure={structure_test}"
        # Note: Printing from parallel workers might result in interleaved output
        print(f"\n[Parallel] Fold {i+1}: {condition_name}")
    
    try:
        unique_patterns_train = unique_pattern_analysis(train_df, verbose=False)
            
        # Use warm_start_u as initial guess
        res = minimize(
            fun=negative_log_likelihood,
            x0=warm_start_u,
            args=(unique_patterns_train, inference_method, param_names, fixed_settings),
            method=method,
            options={'disp': False, 'maxiter': 1000}
        )

        train_nll = res.fun
        fitted_params = _params_from_reduced_unconstrained(res.x, param_names, fixed_settings)
        fit_success = res.success
        optimized_u = res.x
        
        # Calculate Test NLL
        unique_patterns_test = unique_pattern_analysis(test_df, verbose=False)
        
        test_nll = negative_log_likelihood(
            params=optimized_u,
            unique_patterns=unique_patterns_test,
            inference_method=inference_method,
            param_names=param_names,
            fixed_settings=fixed_settings,
            verbose=False
        )

        return {
            'condition': test_condition,
            'train_nll': train_nll,
            'test_nll': test_nll,
            'fitted_params': fitted_params,
            'success': fit_success,
            'fold_idx': i
        }
    except Exception as e:
        if verbose:
            condition_name = f"Order={order_test}, Question={question_test}, Structure={structure_test}"
            print(f"\n[Parallel] Fold {i+1}: {condition_name} - Training failed: {e}")
        return {
            'condition': test_condition,
            'train_nll': np.inf,
            'test_nll': np.inf,
            'fitted_params': None,
            'success': False,
            'fold_idx': i,
            'error': str(e)
        }

def cross_validate_models_parallel(df, inference_method, param_names=None, fixed_settings={}, method='Powell', full_fit_name=None, n_jobs=2, verbose=True):
    """
    Parallel version of cross_validate_models.
    Perform leave-one-condition-out cross-validation for counterfactual inference models.
    """
    if param_names is None:
        param_names = ['s', 'p_keep', 'br', 'beta_A', 'temperature']

    # Determine free parameters and setup warm start
    free_param_names = [n for n in param_names if n not in fixed_settings]
    n_free = len(free_param_names)

    warm_start_u = np.random.randn(n_free) if n_free > 0 else np.array([])
    initial_u = warm_start_u.copy()

    # Warm start logic (load existing fit)
    if full_fit_name:
        warm_start_path = os.path.join('results', 'exp2', full_fit_name)
        if os.path.exists(warm_start_path):
            if verbose:
                print(f"Loading precomputed fit from {warm_start_path} for warm start...")
            try:
                with open(warm_start_path, 'rb') as f:
                    fit_data = pickle.load(f)
                
                # Extract x
                if 'optimization_result' in fit_data:
                    opt_res = fit_data['optimization_result']
                    loaded_u = opt_res['x'] if isinstance(opt_res, dict) else opt_res.x
                    
                    if len(loaded_u) == n_free:
                        warm_start_u = loaded_u
                    else:
                        if verbose:
                            print(f"Warm start parameter count mismatch.")
            except Exception as e:
                if verbose:
                    print(f"Error loading warm start file: {e}")

    # Compute full fit if needed
    is_loaded = not np.array_equal(warm_start_u, initial_u) if n_free > 0 else True
    
    if full_fit_name and not is_loaded and n_free > 0:
        if verbose:
            print("Running full dataset fit to initialize cross-validation folds (Warm Start)...")
        
        full_fit_result = fit_model(
            df=df,
            inference_method=inference_method,
            param_names=param_names,
            fixed_settings=fixed_settings,
            method=method,
            verbose=verbose
        )
        
        if full_fit_result['success'] or full_fit_result['optimization_result'].success:
             warm_start_u = full_fit_result['optimization_result'].x
        
        try:
            warm_start_path = os.path.join('results', 'exp2', full_fit_name)
            os.makedirs(os.path.dirname(warm_start_path), exist_ok=True)
            with open(warm_start_path, 'wb') as f:
                pickle.dump(full_fit_result, f)
            if verbose:
                print(f"Saved warm start fit to {warm_start_path}")
        except Exception as e:
            print(f"Error saving warm start: {e}")

    # Define all 8 conditions
    conditions = []
    for order in [1, 2]:  # ACD, DCA
        for question in [0, 1]:  # off, on
            for structure in [1, 2]:  # disjunctive, conjunctive
                conditions.append((order, question, structure))
    
    if verbose:
        print(f"Parallel cross-validation with {len(conditions)} conditions")
        print("-" * 60)
    
       
    # Prepare tasks
    tasks = []
    for i, test_condition in enumerate(conditions):
        order_test, question_test, structure_test = test_condition
        
        # Create train/test splits
        test_mask = ((df['order'] == order_test) & 
                    (df['question'] == question_test) & 
                    (df['structure'] == structure_test))
        
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        
        tasks.append((i, test_condition, train_df, test_df))
        
    # Run parallel execution (passing warm_start_u)
    fold_results = Parallel(n_jobs=n_jobs, verbose= 1 if verbose else 0)(
        delayed(_fit_single_condition_fold)(
            i, test_condition, train_df, test_df, warm_start_u, inference_method, 
            param_names, fixed_settings, method, verbose
        )
        for i, test_condition, train_df, test_df in tasks
    )
    
    # Sort by fold index to identify conditions correctly
    fold_results = sorted(fold_results, key=lambda x: x['fold_idx'])
    
    method_results = {
        'conditions': [r['condition'] for r in fold_results],
        'train_nlls': [r['train_nll'] for r in fold_results],
        'test_nlls': [r['test_nll'] for r in fold_results],
        'fitted_params': [r['fitted_params'] for r in fold_results],
        'success_rates': [r['success'] for r in fold_results]
    }
    
    return method_results
