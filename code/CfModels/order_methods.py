import numpy as np
from .inference import CounterfactualInferenceBN
from .networks import create_diamond_bayesian_network
from .utils import binary_softmax

from itertools import product



def static_twin(
        base_inf,
        disjunction,
        path,
        obs,
        cf_constraint,
        s,
        temperature=None,
        p_keep=1.0,
        compute_exogenous=False,
        smoothing = 'exponential',
        verbose=False):
    """
    Static analytic function for counterfactual inference in Bayesian networks.
    """
    
    def apply_temp(p):
        if temperature is None:
            return p
        return binary_softmax(p, temperature, method=smoothing)

    # Order details
    order_vars = [list(p.keys())[0] for p in path]   # [q1,q2,q3]
    path_vals  = [list(p.values())[0] for p in path] # [v1,v2,v3]
    q1, q2, q3 = order_vars
    v1, v2, v3 = path_vals
    exog = base_inf.exogenous_vars

    # Check if p_keep = 0, if so query all ACD variables jointly and take marginal inferences
    if p_keep == 0.0:
        ps = base_inf.query(variables=[q1, q2, q3], evidence=obs, cf_evidence=cf_constraint, marginals=1, s=s)

        p1 = ps[q1]
        p1_decision = apply_temp(p1)
        if v1 is None:
            P1 = p1
        else:
            P1 = p1_decision if v1 == 1 else (1 - p1_decision)
        
        p2 = ps[q2]
        p2_decision = apply_temp(p2)
        if v2 is None:
            P2 = p2
        else:
            P2 = p2_decision if v2 == 1 else (1 - p2_decision) 
        
        p3 = ps[q3]
        p3_decision = apply_temp(p3)
        if v3 is None:
            P3 = p3
        else:
            P3 = p3_decision if v3 == 1 else (1 - p3_decision)
        
        result = P1 * P2 * P3
        
        if not compute_exogenous:
            return {'path': path_vals, 
                    'probability': result,
                    'P1': P1,
                    'P2': P2,
                    'P3': P3,
                    'method': 'static'
            }
        
        # Handle exogenous case for p_keep = 0
        prior = base_inf.query(variables=list(exog), marginals=1, s=0)
        
        # All use only cf_constraint
        exog_q =  base_inf.query(variables=list(exog), evidence=obs, cf_evidence=cf_constraint, marginals=1, s=s)
        
        return {
            'path': path_vals,
            'probability': result,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'exogenous_marginals': {
                'prior': prior,
                'marginals_q1': exog_q,
                'marginals_q2': exog_q,
                'marginals_q3': exog_q
            },
            'method': 'static'
        }

    # 1) First judgement probability (fixed)
    p1 = base_inf.query(variables=[q1], evidence=obs, cf_evidence=cf_constraint, s=s).values[1]
    p1_decision = apply_temp(p1)
    if v1 is None:
        P1 = p1
    else:
        P1 = p1_decision if v1 == 1 else (1 - p1_decision)

    # Skip unnecessary calculations if p_keep == 1
    if p_keep == 1.0:
        # When p_keep=1, only need K1->K2->K3 path
        E_cf_q1 = {**cf_constraint, q1: v1}
        E_cf_q1_q2 = {**cf_constraint, q1: v1, q2: v2}
        
        p2_K1 = base_inf.query(variables=[q2], evidence=obs, cf_evidence=E_cf_q1, s=s).values[1]
        p2_K1_decision = apply_temp(p2_K1)
        if v2 is None:
            P2_K1 = p2_K1
        else:
            P2_K1 = p2_K1_decision if v2 == 1 else (1 - p2_K1_decision)
        
        p3_K1K2 = base_inf.query(variables=[q3], evidence=obs, cf_evidence=E_cf_q1_q2, s=s).values[1]
        p3_K1K2_decision = apply_temp(p3_K1K2)
        if v3 is None:
            P3_K1K2 = p3_K1K2 
        else:
            P3_K1K2 = p3_K1K2_decision if v3 == 1 else (1 - p3_K1K2_decision)
        
        result = P1 * P2_K1 * P3_K1K2
        result_1 = P1
        result_2 = P2_K1
        result_3 = P3_K1K2
    
        if not compute_exogenous:
            return {'path': path_vals, 
                    'probability': result,
                    'P1': result_1,
                    'P2': result_2,
                    'P3': result_3,
                    'method': 'static'
            }
    

        # Calculate step exogenous probabilities for comparison
        prior = base_inf.query(variables=list(exog), marginals=1, s=0)

        # Step 1: exogenous after committing to q1=v1
        exog_q1 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1}, marginals=1, s=s)
        
        # Step 2: exogenous after committing to q2=v2
        exog_q2 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1, q2: v2}, marginals=1, s=s)

        # Step 3: exogenous after committing to q3=v3
        exog_q3 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1, q2: v2, q3: v3}, marginals=1, s=s)
        

        # Return result with exogenous marginals
        return {'path': path_vals, 
                'probability': result,
                'P1': result_1,
                'P2': result_2,
                'P3': result_3,
                'exogenous_marginals': {
                    'prior': prior,
                    'marginals_q1': exog_q1,
                    'marginals_q2': exog_q2,
                    'marginals_q3': exog_q3
                },
                'method': 'static'
        }

    else:
    
        # 2) Second judgement probabilities under K1 vs ~K1
        #    Infer with cf + {q1=v1} (keep) or cf only (drop)
        E_cf       = {**cf_constraint}
        E_cf_q1    = {**cf_constraint, q1: v1}
        
        # Remove any None values from the dictionaries
        
        # Chained inference for K1
        p2_K1 = base_inf.query(variables=[q2], evidence=obs, cf_evidence=E_cf_q1, s=s).values[1]
        p2_K1_decision = apply_temp(p2_K1)
        # Chained inference for ~K1
        p2_nK1 = base_inf.query(variables=[q2], evidence=obs, cf_evidence=E_cf, s=s).values[1]
        p2_nK1_decision = apply_temp(p2_nK1)
        if v2 is None:
            P2_nK1 = p2_nK1
            P2_K1 = p2_K1
        else:
            P2_nK1 = p2_nK1_decision if v2 == 1 else (1 - p2_nK1_decision)
            P2_K1 = p2_K1_decision if v2 == 1 else (1 - p2_K1_decision)

        # 3) Third judgement probabilities for the 4 cases:
        #    Using chained inference again with evidence sets:
        #      K1,K2:   cf + {q1=v1, q2=v2}
        #      K1,~K2:  cf + {q1=v1}
        #      ~K1,K2:  cf + {q2=v2}
        #      ~K1,~K2: cf
        E_K1K2  = {**cf_constraint, q1: v1, q2: v2}
        E_K1nK2 = {**cf_constraint, q1: v1}
        E_nK1K2 = {**cf_constraint, q2: v2}
        E_nK1nK2= {**cf_constraint}

        # K1,K2
        p3_K1K2 = base_inf.query(variables=[q3], evidence=obs, cf_evidence=E_K1K2, s=s).values[1]
        p3_K1K2_decision = apply_temp(p3_K1K2)
        # K1,~K2
        p3_K1nK2 = base_inf.query(variables=[q3], evidence=obs, cf_evidence=E_K1nK2, s=s).values[1]
        p3_K1nK2_decision = apply_temp(p3_K1nK2)
        # ~K1,K2
        p3_nK1K2 = base_inf.query(variables=[q3], evidence=obs, cf_evidence=E_nK1K2, s=s).values[1]
        p3_nK1K2_decision = apply_temp(p3_nK1K2)
        # ~K1,~K2
        p3_nK1nK2 = base_inf.query(variables=[q3], evidence=obs, cf_evidence=E_nK1nK2, s=s).values[1]
        p3_nK1nK2_decision = apply_temp(p3_nK1nK2)
        
        if v3 is None:
            P3_K1K2 = p3_K1K2
            P3_K1nK2 = p3_K1nK2 
            P3_nK1K2 = p3_nK1K2
            P3_nK1nK2 = p3_nK1nK2
        else:
            P3_K1K2 = p3_K1K2_decision if v3 == 1 else (1 - p3_K1K2_decision)
            P3_K1nK2 = p3_K1nK2_decision if v3 == 1 else (1 - p3_K1nK2_decision)
            P3_nK1K2 = p3_nK1K2_decision if v3 == 1 else (1 - p3_nK1K2_decision)
            P3_nK1nK2 = p3_nK1nK2_decision if v3 == 1 else (1 - p3_nK1nK2_decision)

        # Mixture over keep/drop outcomes
        p = p_keep
        q = 1 - p
        mix = (
            p * (P2_K1)   * (p * P3_K1K2   + q * P3_K1nK2) +
            q * (P2_nK1)  * (p * P3_nK1K2  + q * P3_nK1nK2)
        )
        result = P1 * mix
        result_1 = P1
        result_2 = (p * P2_K1 + q * P2_nK1)
        result_3 = (p * P3_K1K2   + q * P3_K1nK2)
    if verbose:
        print(f"P1={P1:.6f}, P2_K1={P2_K1:.6f}, P2_nK1={P2_nK1:.6f}")
        print(f"P3_K1K2={P3_K1K2:.6f}, P3_K1nK2={P3_K1nK2:.6f}, P3_nK1K2={P3_nK1K2:.6f}, P3_nK1nK2={P3_nK1nK2:.6f}")
        print(f"Analytic path probability = {result:.6f}")

    if not compute_exogenous:
        return {'path': path_vals, 
                'probability': result,
                'P1': result_1,
                'P2': result_2,
                'P3': result_3,
                'method': 'static'
        }

    # Calculate step exogenous probabilities for comparison
    prior = base_inf.query(variables=list(exog), marginals=1, s=0)

    # Step 1: exogenous after committing to q1=v1
    exog_q1 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1}, marginals=1, s=s)

    # Step 2: weighted average for K1 and ~K1 cases
    exog_q2_k1 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1, q2: v2}, marginals=1, s=s)
    exog_q2_nk1 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q2: v2}, marginals=1, s=s)
    exog_q2 = {
        var: p_keep * exog_q2_k1[var] + (1 - p_keep) * exog_q2_nk1[var]
        for var in exog
    }

    # Step 3: weighted average for all four cases
    exog_q3_k1k2 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1, q2: v2, q3: v3}, marginals=1, s=s)
    exog_q3_k1nk2 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q1: v1, q3: v3}, marginals=1, s=s)
    exog_q3_nk1k2 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q2: v2, q3: v3}, marginals=1, s=s)
    exog_q3_nk1nk2 = base_inf.query(variables=list(exog), evidence=obs, cf_evidence={**cf_constraint, q3: v3}, marginals=1, s=s)

    p = p_keep
    exog_q3 = {
        var: (p * p) * exog_q3_k1k2[var] +
             (p * (1 - p)) * exog_q3_k1nk2[var] +
             ((1 - p) * p) * exog_q3_nk1k2[var] +
             ((1 - p) * (1 - p)) * exog_q3_nk1nk2[var]
        for var in exog
    }

    # Return result with exogenous marginals
    return {'path': path_vals, 
            'probability': result,
            'P1': result_1,
            'P2': result_2,
            'P3': result_3,
            'exogenous_marginals': {
                'prior': prior,
                'marginals_q1': exog_q1,
                'marginals_q2': exog_q2,
                'marginals_q3': exog_q3
            },
            'method': 'static'
    }

def static_twin_with_missing(base_inf, disjunction, path, obs, cf_constraint, s, temperature=None, p_keep=1.0, smoothing = 'exponential', compute_exogenous=False, verbose=False):
    """
    Static analytic function that handles missing values in the path by averaging over all possible completions.
    """

    # Check if any path elements contain None
    has_missing = any(list(step.values())[0] is None for step in path)
    
    if not has_missing:
        # No missing values, call original function
        return static_twin(base_inf, disjunction, path, obs, cf_constraint, s, temperature, p_keep, smoothing, compute_exogenous, verbose)
    
    # Find positions with None values
    none_positions = []
    for i, step in enumerate(path):
        if list(step.values())[0] is None:
            none_positions.append(i)
    
    # Generate all possible combinations for None positions
    n_missing = len(none_positions)
    combinations = list(product([0, 1], repeat=n_missing))
    
    total_prob = 0.0
    n_combinations = len(combinations)
    
    for combination in combinations:
        # Create new path with None values replaced
        new_path = []
        for i, step in enumerate(path):
            var_name = list(step.keys())[0]
            var_value = list(step.values())[0]
            
            if var_value is None:
                # Replace None with value from current combination
                none_idx = none_positions.index(i)
                new_value = combination[none_idx]
                new_path.append({var_name: new_value})
            else:
                new_path.append(step)
        
        # Call original function with complete path
        result = static_twin(base_inf, disjunction, new_path, obs, cf_constraint, s, temperature, p_keep, smoothing, compute_exogenous, verbose)
        total_prob += result['probability']
    
    # Average the probabilities
    avg_prob = total_prob / n_combinations
    
    return {
        'path': [list(step.values())[0] for step in path],
        'probability': avg_prob,
        'method': 'static_with_missing'
    }
