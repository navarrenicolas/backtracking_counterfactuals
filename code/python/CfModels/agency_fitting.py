import numpy as np

from scipy.optimize import minimize

from .utils import *
from .inference import CounterfactualInferenceBN
from .networks import create_diamond_bayesian_network

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
    br = transformed['br']

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
    
    
    beta_B = br
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
    required_params = {
        'beta_A': beta_A,
        'beta_B': beta_B,
        'beta_C': beta_C,
        'beta_D': beta_D,
        'theta_AB': theta_AB,
        'theta_AC': theta_AC,
        'theta_BD': theta_BD,
        'theta_CD': theta_CD
    }
    
    none_params = [name for name, value in required_params.items() if value is None]
    if none_params:
        raise ValueError(f"The following required parameters are None: {', '.join(none_params)}")

    # Build models with current parameters (command/outcome powers and base rates may be in fixed_settings)
    disj_model = create_diamond_bayesian_network(
        power_B=theta_AB,
        power_C=theta_AC,
        power_D_from_B=theta_BD,
        power_D_from_C=theta_CD,
        beta_A=beta_A,
        beta_B=beta_B,
        beta_C=beta_C,
        beta_D=beta_D,
        disjunction= True
    )

    conj_model = create_diamond_bayesian_network(
        power_B=theta_AB,
        power_C=theta_AC,
        power_D_from_B=theta_BD,
        power_D_from_C=theta_CD,
        beta_A=beta_A,
        beta_B=beta_B,
        beta_C=beta_C,
        beta_D=beta_D,
        disjunction= False
    )

    inference_engines = {
        'disj_inference': CounterfactualInferenceBN(disj_model),
        'conj_inference': CounterfactualInferenceBN(conj_model)
    } 

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


###################
## CROSS VALIDATION
###################

def cross_validate_models(df, model_data, method='L-BFGS-B', verbose=True):
    """
    Perform leave-one-condition-out cross-validation for counterfactual inference models.
    
    There are 8 conditions of interest:
    - order (ACD=1/DCA=2) × question (off=0/on=1) × structure (disjunctive=1/conjunctive=2)
    
    Args:
        df (pd.DataFrame): Full dataset with columns ['order', 'question', 'structure', 'A', 'C', 'D']
        inference_methods (dict): Dictionary mapping method names to inference functions
                                 e.g., {'static': static_analytic_with_missing, 'abduction': abduction_analytic_with_missing}
        param_names (list): Parameter names to fit (default: ['s', 'p_keep', 'br', 'beta_A', 'temperature'])
        fixed_settings (dict): Parameters to fix during fitting
        method (str): Optimization method for scipy.minimize
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Cross-validation results with train/test NLL for each condition and method
    """
    # Define all 8 conditions
    conditions = []
    for order in [1, 2]:  # ACD, DCA
        for question in [0, 1]:  # off, on
            for structure in [1, 2]:  # disjunctive, conjunctive
                conditions.append((order, question, structure))
    
    if verbose:
        print(f"Cross-validation with {len(conditions)} conditions")
        print(f"Models to evaluate: {list(model_data.keys())}")
        print("-" * 60)
    
    # Store results for each method
    cv_results = {}
    # for model_name, mdata in reversed(list(model_data.items())[-2:]):
    for model_name, mdata  in model_data.items():
        if verbose:
            print(f"\n=== Cross-validating {model_name.upper()} method ===")
        
        inference_method = mdata['inference_method']
        param_names = mdata.get('param_names', [])
        fixed_settings = mdata.get('fixed_settings', {})

        method_results = {
            'conditions': [],
            'train_nlls': [],
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
                    print('param names:', param_names)
                    print('fixed settings:', fixed_settings)
                fit_result = fit_model(
                    df=train_df,
                    inference_method=inference_method,
                    param_names=param_names,
                    fixed_settings=fixed_settings,
                    method=method,
                    verbose=False
                )
                
                train_nll = fit_result['nll']
                fitted_params = fit_result['fitted_params']
                fit_success = fit_result['success']
                
                if verbose:
                    print(f"  Train NLL: {train_nll:.4f}, Success: {fit_success}")
            
                    
            except Exception as e:
                if verbose:
                    print(f"  Training failed: {e}")
                train_nll = np.inf
                fitted_params = None
                fit_success = False
            
            # Store results
            method_results['conditions'].append(test_condition)
            method_results['train_nlls'].append(train_nll)
            method_results['fitted_params'].append(fitted_params)
            method_results['success_rates'].append(fit_success)
        
    
        cv_results[model_name] = method_results

    return cv_results


