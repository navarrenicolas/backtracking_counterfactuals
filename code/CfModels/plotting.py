import matplotlib.pyplot as plt
import numpy as np

from .networks import create_diamond_bayesian_network
from .inference import CounterfactualInferenceBN
from .order_methods import static_twin 
from .fitting import make_priors

def plot_exogenous_evolution(order_effects_result, exog_vars):
    """
    Plots the evolution of exogenous variables for all paths in the order_effects result.
    
    Args:
        order_effects_result (dict): Output from order_effects function containing 'results' list.
                                     Each result should have 'exog_evolution' (list of dicts).
        exog_vars (list): List of exogenous variable names to plot.
    """
    results = order_effects_result.get('results', [])
    if not results:
        print("No results found in order_effects_result")
        return

    order = order_effects_result.get('order', 'Unknown')
    
    # Create plot grid (2 rows x 4 columns for 8 paths)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    cmap = plt.cm.RdYlBu_r
    im = None
    
    for idx, result in enumerate(results):
        if idx >= len(axes): break
        
        ax = axes[idx]
        path_values = result['path']
        path_vars = [c for c in order]
        
        # Construct path label
        path_steps = [f"{var}={val}" for var, val in zip(path_vars, path_values)]
        path_label = ", ".join(path_steps)
        
        # Extract exogenous evolution data
        # Expecting 'exog_evolution' to be a list of dicts: [Prior, Step1, Step2, Step3]
        exog_data = result.get('exogenous_marginals', [])
        
        if not exog_data:
            ax.text(0.5, 0.5, "No exogenous data", ha='center', va='center')
            ax.set_title(f"Path: {path_label}")
            continue
            
        # Prepare matrix for heatmap
        # Rows: variables, Cols: steps
        # Data is organized as a dictionary for each step
        # exog_data['prior'], exog_data['marginals_q1'], exog_data['marginals_q2'], exog_data['marginals_q3']
        
        # get the columns in order
        prior_marginals = exog_data['prior']  # prior
        marginals_q1 = exog_data['marginals_q1']  # after first query
        marginals_q2 = exog_data['marginals_q2']  # after second query
        marginals_q3 = exog_data['marginals_q3']  # after third query

        matrix_data = []
        for var in exog_vars:
            row = [
                prior_marginals.get(var, 0.0),
                marginals_q1.get(var, 0.0),
                marginals_q2.get(var, 0.0),
                marginals_q3.get(var, 0.0)
            ]
            matrix_data.append(row)

        matrix = np.array(matrix_data)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Labels
        ax.set_title(f"Path: {path_label}", fontsize=10, fontweight='bold')
        ax.set_yticks(range(len(exog_vars)))
        ax.set_yticklabels(exog_vars)
        
        step_labels = ['Prior'] + [f'Step {i+1}' for i in range(matrix.shape[1]-1)]
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(step_labels, rotation=45, ha='right')
        
        # Annotations
        for i in range(len(exog_vars)):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                # Choose text color based on background intensity
                text_color = "white" if (val < 0.3 or val > 0.7) else "black"
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=8)

    # Global title
    fig.suptitle(f'Exogenous Variable Evolution - Order {order}', fontsize=16, fontweight='bold')
    
    # Colorbar
    if im:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Probability')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def analyze_ordering_effects(inference_method,beta_A=0.5, beta_B=0.5, beta_C=0.5, beta_D=0.5,
                                   power_B=0.5, power_C=0.5, power_D_from_B=0.5, power_D_from_C=0.5,
                                   disjunction=True, s=0.5, obs={'A': 1, 'B': 1, 'C': 1, 'D': 1}, 
                                   cf_constraint={'B': 0}, temperature=None, p_keep=1.0, verbose=False):
    """
    Analyze ordering effects using the static_twin function for clean computation.
    
    Args:
        beta_A, beta_B, beta_C, beta_D (float): Base rates for each variable
        power_B, power_C (float): Causal powers from A to B and C
        power_D_from_B, power_D_from_C (float): Causal powers from B and C to D
        disjunction (bool): If True, D is disjunction of B and C; if False, conjunction
        s (float): Stability value for counterfactual inference
        obs (dict): Observations of real variables
        cf_constraint (dict): Initial counterfactual constraints
        temperature (float): Temperature parameter for softmax
        p_keep (float): Probability of keeping intermediate commitments
        verbose (bool): If True, print debug information
    
    Returns:
        dict: Results containing ACD and DCA probabilities for A, C, D
    """
    
    # Create diamond Bayesian network
    diamond = create_diamond_bayesian_network(
        disjunction=disjunction,
        beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
        power_B=power_B, power_C=power_C, 
        power_D_from_B=power_D_from_B, power_D_from_C=power_D_from_C,
        
    )
    
    # Create inference engine
    cf_inference = CounterfactualInferenceBN(diamond)
    
    # Generate all possible paths for ACD and DCA orderings
    ACD_paths = []
    for a_val in [0, 1]:
        for c_val in [0, 1]:
            for d_val in [0, 1]:
                path = [{'A': a_val}, {'C': c_val}, {'D': d_val}]
                ACD_paths.append(path)

    DCA_paths = []
    for d_val in [0, 1]:
        for c_val in [0, 1]:
            for a_val in [0, 1]:
                path = [{'D': d_val}, {'C': c_val}, {'A': a_val}]
                DCA_paths.append(path)
    
    # ACD ordering - compute path probabilities using static_twin
    ACD_results = []
    for path in ACD_paths:
        result = inference_method(
            base_inf=cf_inference,
            disjunction=disjunction,
            path=path,
            obs=obs,
            cf_constraint=cf_constraint,
            s=s,
            temperature=temperature,
            p_keep=p_keep,
            verbose=verbose
        )
        ACD_results.append(result)
    
    # DCA ordering - compute path probabilities using inference method  
    DCA_results = []
    for path in DCA_paths:
        result = inference_method(
            base_inf=cf_inference,
            disjunction=disjunction,
            path=path,
            obs=obs,
            cf_constraint=cf_constraint,
            s=s,
            temperature=temperature,
            p_keep=p_keep,
            verbose=verbose
        )
        DCA_results.append(result)
    
    # Compute marginal probabilities for ACD ordering
    total_A_prob_ACD = sum(result['probability'] * result['path'][0] for result in ACD_results)
    total_C_prob_ACD = sum(result['probability'] * result['path'][1] for result in ACD_results)
    total_D_prob_ACD = sum(result['probability'] * result['path'][2] for result in ACD_results)
    
    # Compute marginal probabilities for DCA ordering
    total_D_prob_DCA = sum(result['probability'] * result['path'][0] for result in DCA_results)
    total_C_prob_DCA = sum(result['probability'] * result['path'][1] for result in DCA_results)
    total_A_prob_DCA = sum(result['probability'] * result['path'][2] for result in DCA_results)
    

    return {
        'A_ACD': total_A_prob_ACD,
        'A_DCA': total_A_prob_DCA,
        'C_ACD': total_C_prob_ACD,
        'C_DCA': total_C_prob_DCA,
        'D_ACD': total_D_prob_ACD,
        'D_DCA': total_D_prob_DCA
    }

def order_effects(beta_A=0.5, beta_B=0.5, beta_C=0.5, beta_D=0.5,
                  power_B=0.5, power_C=0.5, power_D_from_B=0.5, power_D_from_C=0.5,
                  disjunction=True, 
                  question = True,
                  order='ACD',
                  s=0.5,
                  temperature=None,
                  p_keep=1,
                  inference_method=static_twin,
                  verbose=False):
    """
    Analyze ordering effects using abduction approach with optional stability retention.
    
    Args:
        beta_A, beta_B, beta_C, beta_D (float): Base rates for each variable
        power_B, power_C (float): Causal powers from A to B and C
        power_D_from_B, power_D_from_C (float): Causal powers from B and C to D
        disjunction (bool): If True, D is disjunction of B and C; if False, conjunction
        s (float): Stability value for counterfactual inference
        order (str): 'ACD' or 'DCA' ordering to analyze
        temperature (float): Temperature parameter for softmax (if applicable)
        destabilize (bool): If True, apply destabilization to exogenous variables
        p_keep (float): Probability of keeping previous judgements
        method (str): 'abduction' or 'dynamic_commitment' method to use
        verbose (bool): If True, print detailed information
    Returns:
        dict: Results containing probabilities for A, C, D based on specified ordering
    """


    if not question:
        obs = {'A':1,'B':1,'C':1,'D':1}
        cf_constraint = {'B':0}        
    else:
        obs = {'A':0,'B':0,'C':0,'D':0}
        cf_constraint = {'B': 1}
    
        
    diamond = create_diamond_bayesian_network(
        beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
        power_B=power_B, power_C=power_C, 
        power_D_from_B=power_D_from_B, power_D_from_C=power_D_from_C,
        disjunction=disjunction
    )
    diamond_inf = CounterfactualInferenceBN(diamond)
    

    # Define all possible variable orderings
    order_map = {
        'ACD': ['A', 'C', 'D'],
        'DCA': ['D', 'C', 'A'],
        'ADC': ['A', 'D', 'C'],
        'CAD': ['C', 'A', 'D'],
        'CDA': ['C', 'D', 'A'],
        'DAC': ['D', 'A', 'C']
    }
    
    if order not in order_map:
        raise ValueError(f"Unknown order: {order}. Must be one of: {', '.join(order_map.keys())}")
    
    query_vars = order_map[order]

    paths = [
        [{query_vars[0]: q1_val}, {query_vars[1]: q2_val}, {query_vars[2]: q3_val}]
        for q1_val in [0, 1] for q2_val in [0, 1] for q3_val in [0, 1]
    ]

    order_effects = {}
    order_effects['order'] = order
    order_effects['params'] = {
        'beta_A': beta_A, 'beta_B': beta_B, 'beta_C': beta_C, 'beta_D': beta_D,
        'power_B': power_B, 'power_C': power_C,
        'power_D_from_B': power_D_from_B, 'power_D_from_C': power_D_from_C,
        'disjunction': disjunction,
        'question': question,
        's': s,
        'temperature': temperature,
        'p_keep': p_keep
    }
    order_effects['results'] = []

    for path in paths:
        order_effects['results'].append(
            inference_method(
                base_inf=diamond_inf,
                disjunction=disjunction,
                path=path,
                obs=obs,
                cf_constraint=cf_constraint,
                s=s,
                temperature=temperature,
                p_keep=p_keep,
                verbose=verbose
            )
        )

    return order_effects


def compute_mean_predictions(order_effects_results):
    """
    Compute mean predictions for A, C, D from order effects results.
    
    Args:
        order_effects_results (dict): Results from order_effects function with 'results' key
        
    Returns:
        dict: Mean predictions and metadata
    """
    if not order_effects_results or 'results' not in order_effects_results:
        return {'error': 'Invalid results structure'}
    
    results_list = order_effects_results['results']
    if not results_list:
        return {'error': 'Empty results list'}
    
    # Extract metadata
    order = order_effects_results.get('order', 'unknown')
    method = order_effects_results.get('method', 'unknown')
    params = order_effects_results.get('params', {})
    
    # Initialize sums and counts
    A_sum = C_sum = D_sum = 0.0
    total_prob = 0.0
    
    # Compute weighted averages
    for result in results_list:
        path = result['path']
        prob = result['probability']
        total_prob += prob
        
        # Extract values from path based on order
        order_map = {'A': 0, 'C': 0, 'D': 0}
        for i, var in enumerate(order):
            order_map[var] = path[i]
        A_val, C_val, D_val = order_map['A'], order_map['C'], order_map['D']
            
        # Accumulate weighted sums
        A_sum += A_val * prob
        C_sum += C_val * prob
        D_sum += D_val * prob
    
    # Compute means
    if total_prob > 0:
        mean_A = A_sum / total_prob
        mean_C = C_sum / total_prob
        mean_D = D_sum / total_prob
    else:
        mean_A = mean_C = mean_D = 0

    return {
        'order': order,
        'mean_A': mean_A,
        'mean_C': mean_C, 
        'mean_D': mean_D,
        'total_probability': total_prob,
        'parameters': params,
        'method': method
    }


def compare_order(beta_A=0.5, beta_B=0.5, beta_C=0.5, beta_D=0.5,
                  power_B=0.5, power_C=0.5, power_D_from_B=0.5, power_D_from_C=0.5,
                  disjunction=True,
                  question=True,
                  s=0.5,
                  temperature=None,
                  p_keep=.5,
                  inference_method=static_twin,
                  verbose=False):
    """
    Compare order effects between ACD and DCA orderings.
    """
    acd_results = order_effects(
        beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
        power_B=power_B, power_C=power_C, power_D_from_B=power_D_from_B, power_D_from_C=power_D_from_C,
        disjunction=disjunction,
        question=question,
        order='ACD',
        s=s,
        temperature=temperature,
        p_keep=p_keep,
        inference_method=inference_method,
        verbose=verbose
    )
    dca_results = order_effects(
        beta_A=beta_A, beta_B=beta_B, beta_C=beta_C, beta_D=beta_D,
        power_B=power_B, power_C=power_C, power_D_from_B=power_D_from_B, power_D_from_C=power_D_from_C,
        disjunction=disjunction,
        question=question,
        order='DCA',
        s=s,
        temperature=temperature,
        p_keep=p_keep,
        inference_method=inference_method,
        verbose=verbose
    )
    ACD_means = compute_mean_predictions(acd_results)
    DCA_means = compute_mean_predictions(dca_results)

    return ACD_means, DCA_means


def plot_order_means(acd_means, dca_means, title="Comparison of Order Effects"):
    """
    Plot the different order means as a line plot.

    Args:
        acd_means (dict): Dictionary containing mean predictions for ACD ordering
        dca_means (dict): Dictionary containing mean predictions for DCA ordering
        title (str): Title for the plot
    """
    variables = ['A', 'C', 'D']
    ACD_means = [acd_means['mean_A'], acd_means['mean_C'], acd_means['mean_D']]
    DCA_means = [dca_means['mean_A'], dca_means['mean_C'], dca_means['mean_D']]

    plt.figure(figsize=(8, 6))
    plt.plot(variables, ACD_means, 'o-', label='ACD ordering', linewidth=3, markersize=8, color='blue')
    plt.plot(variables, DCA_means, 's-', label='DCA ordering', linewidth=3, markersize=8, color='red')
    plt.ylabel('Mean Probability')
    plt.xlabel('Variables')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_order_means_with_data(acd_means, dca_means, empirical_means=None, title="Comparison of Order Effects"):
    """
    Plot model-predicted order means for ACD and DCA and optionally overlay empirical means with SEM error bars.

    Args:
        acd_means (dict): model predictions for ACD ordering (expects keys 'mean_A','mean_C','mean_D')
        dca_means (dict): model predictions for DCA ordering (same keys as above)
        empirical_means (dict, optional): empirical means in the format:
            {'ACD': {'A': {'mean':..,'sem':..}, 'C': {...}, 'D': {...}}, 'DCA': {...}}
        title (str): plot title
    """
    variables = ['A', 'C', 'D']
    ACD_vals = [acd_means['mean_A'], acd_means['mean_C'], acd_means['mean_D']]
    DCA_vals = [dca_means['mean_A'], dca_means['mean_C'], dca_means['mean_D']]

    plt.figure(figsize=(8, 6))
    x = np.arange(len(variables))
    # Plot model predictions
    plt.plot(x, ACD_vals, 'o-', label='ACD model', linewidth=3, markersize=8, color='black')
    plt.plot(x, DCA_vals, 's-', label='DCA model', linewidth=3, markersize=8, color='grey')

    # Overlay empirical means if provided
    if empirical_means is not None:
        try:
            em_acd = [empirical_means['ACD'][v]['mean'] for v in variables]
            em_acd_sem = [empirical_means['ACD'][v]['sem'] for v in variables]
            em_dca = [empirical_means['DCA'][v]['mean'] for v in variables]
            em_dca_sem = [empirical_means['DCA'][v]['sem'] for v in variables]
            # Plot empirical points with error bars
            plt.errorbar(x, em_acd, yerr=em_acd_sem, fmt='o', color='black', label='ACD empirical', markersize=7, capsize=4, linestyle='None')
            plt.errorbar(x, em_dca, yerr=em_dca_sem, fmt='s', color='grey', label='DCA empirical', markersize=7, capsize=4, linestyle='None')
        except Exception as e:
            print('Could not plot empirical means: ', e)

    plt.xticks(x, variables)
    plt.ylabel('Mean Probability')
    plt.xlabel('Variables')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_mle_order_effects_grid(data, mle_fit, inference_method, title_prefix="Order Effects", verbose=False):
    """Plot ACD vs DCA model predictions and empirical means for all 4 (structure × question)
    pairings in a 2×2 subplot grid.

    Layout:
      rows = structure (top: Disjunctive, bottom: Conjunctive)
      cols = question  (left: Off, right: On)

    Args:
        data (pd.DataFrame): experimental dataframe (expects columns 'structure','question','order','A','C','D')
        mle_fit (dict): result returned by fit_model (must contain 'fitted_params')
        method (str): inference method name passed to compare_order
        title_prefix (str): overall figure title prefix
        verbose (bool): if True print some diagnostics

    Returns:
        dict: mapping (disjunction, question) -> (ACD_means, DCA_means, empirical_means)
    """
    # build prior from mle_fit (same logic as existing single-panel function)
    fitted = mle_fit.get('fitted_params', {})
    nll = mle_fit.get('nll', None)
    

    s = fitted['s']
    p_keep = fitted['p_keep']
    temperature = fitted['temperature']
   
    prior = make_priors(fitted)
    # rename powerrs to match function signature
    prior['power_B'] = prior.pop('theta_AB')
    prior['power_C'] = prior.pop('theta_AC')
    prior['power_D_from_B'] = prior.pop('theta_BD')
    prior['power_D_from_C'] = prior.pop('theta_CD')

    # prepare figure: rows = [Disjunctive, Conjunctive], cols = [Question Off, Question On]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.suptitle(title_prefix +  f" nll: {nll:.2f}", fontsize=14)
    variables = ['A', 'C', 'D']
    results = {}

    for i, disjunction in enumerate([True, False]):        # row: True -> Disjunctive (top), False -> Conjunctive (bottom)
        for j, question in enumerate([False, True]):      # col: False -> Off (left), True -> On (right)
            ax = axes[i, j]
            if verbose:
                print(f"Computing panel: disjunction={disjunction}, question={question}")
            # compute model means for this setting
            ACD_means, DCA_means = compare_order(
                **prior,
                disjunction=disjunction,
                s=s,
                temperature=temperature,
                p_keep=p_keep,
                inference_method=inference_method,
                question=question,
                verbose=False
            )

            # assemble empirical means for this cell (same logic as single-panel function)
            empirical_means = {}
            struct_code = [2, 1][int(disjunction)]  # mapping used elsewhere in notebook
            filtered_data = data[(data['structure'] == struct_code) & (data['question'] == int(question))]
            response_map = {-1: 0, 1: 1}
            for order_val in [1, 2]:
                order_data = filtered_data[filtered_data['order'] == order_val]
                order_name = 'ACD' if order_val == 1 else 'DCA'
                empirical_means[order_name] = {}
                for var in variables:
                    responses = order_data[var]
                    valid_responses = responses[responses != 0]  # remove 0-coded non-responses
                    try:
                        converted_responses = [response_map[r] for r in valid_responses]
                    except Exception:
                        converted_responses = []
                    if len(converted_responses) > 0:
                        mean_val = np.mean(converted_responses)
                        sem_val = np.std(converted_responses, ddof=0) / np.sqrt(len(converted_responses))
                        empirical_means[order_name][var] = {'mean': mean_val, 'sem': sem_val, 'n': len(converted_responses)}
                    else:
                        empirical_means[order_name][var] = {'mean': 0.0, 'sem': 0.0, 'n': 0}

            # plot model predictions
            x = np.arange(len(variables))
            ACD_vals = [ACD_means['mean_A'], ACD_means['mean_C'], ACD_means['mean_D']]
            DCA_vals = [DCA_means['mean_A'], DCA_means['mean_C'], DCA_means['mean_D']]
            ax.plot(x -0.06 , ACD_vals, 'o-', label='ACD model', linewidth=2, markersize=6, color='black')
            ax.plot(x +0.06 , DCA_vals, 's-', label='DCA model', linewidth=2, markersize=6, color='grey')

            # overlay empirical means with SEM if available
            try:
                em_acd = [empirical_means['ACD'][v]['mean'] for v in variables]
                em_acd_sem = [empirical_means['ACD'][v]['sem'] for v in variables]
                em_dca = [empirical_means['DCA'][v]['mean'] for v in variables]
                em_dca_sem = [empirical_means['DCA'][v]['sem'] for v in variables]

                # Only plot empirical points if at least one observation exists
                if any(empirical_means['ACD'][v]['n'] > 0 for v in variables):
                    ax.errorbar(x - 0.06, em_acd, yerr=em_acd_sem, fmt='o', color='black', label='ACD empirical', markersize=5, capsize=3, linestyle='None')
                if any(empirical_means['DCA'][v]['n'] > 0 for v in variables):
                    ax.errorbar(x + 0.06, em_dca, yerr=em_dca_sem, fmt='s', color='grey', label='DCA empirical', markersize=5, capsize=3, linestyle='None')
            except Exception:
                # fall back silently if empirical plotting fails for this panel
                pass

            ax.set_xticks(x)
            ax.set_xticklabels(variables)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            panel_title = f"{'Disjunctive' if disjunction else 'Conjunctive'} - Question {'On' if question else 'Off'}" 
            ax.set_title(panel_title, fontsize=11)

            # Only show legend in the top-right panel to reduce clutter
            if i == 0 and j == 1:
                ax.legend(loc='upper right', fontsize=9)
            results[(disjunction, question)] = (ACD_means, DCA_means, empirical_means)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return 

