from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def create_diamond_bayesian_network(power_B=0.9, power_C=0.9, power_D_from_B=0.9, power_D_from_C=0.9, beta_A=0.5, beta_B=0.05, beta_C=0.05, beta_D=0.05, disjunction=True):
    """
    Creates a diamond Bayesian Network using PGMPY that mirrors the SCM implementation.
    
    Args:
        power_B (float): Causal power of A on B (theta_A_B)
        power_C (float): Causal power of A on C (theta_A_C)  
        power_D_from_B (float): Causal power of B on D (theta_B_D)
        power_D_from_C (float): Causal power of C on D (theta_C_D)
        beta_A (float): Base rate for A
        beta_B (float): Base rate for B
        beta_C (float): Base rate for C
        beta_D (float): Base rate for D
        disjunction (bool): If True, D is disjunction of B and C; if False, conjunction
    
    Returns:
        tuple: (DiscreteBayesianNetwork model, VariableElimination inference engine)
    """
    
    # Define the network structure including exogenous variables
    model = DiscreteBayesianNetwork([
        # Exogenous variables for base rates
        ('beta_A', 'A'),
        ('beta_B', 'B'), 
        ('beta_C', 'C'),
        ('beta_D', 'D'),
        
        # Exogenous variables for causal powers
        ('theta_A_B', 'B'),
        ('theta_A_C', 'C'),
        ('theta_B_D', 'D'),
        ('theta_C_D', 'D'),
        
        # Structural edges between endogenous variables
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('C', 'D')
    ])
    
    # CPDs for exogenous variables (Bernoulli distributions)
    cpd_beta_A = TabularCPD('beta_A', 2, [[1-beta_A], [beta_A]])
    cpd_beta_B = TabularCPD('beta_B', 2, [[1-beta_B], [beta_B]])
    cpd_beta_C = TabularCPD('beta_C', 2, [[1-beta_C], [beta_C]])
    cpd_beta_D = TabularCPD('beta_D', 2, [[1-beta_D], [beta_D]])
    
    cpd_theta_A_B = TabularCPD('theta_A_B', 2, [[1-power_B], [power_B]])
    cpd_theta_A_C = TabularCPD('theta_A_C', 2, [[1-power_C], [power_C]])
    cpd_theta_B_D = TabularCPD('theta_B_D', 2, [[1-power_D_from_B], [power_D_from_B]])
    cpd_theta_C_D = TabularCPD('theta_C_D', 2, [[1-power_D_from_C], [power_D_from_C]])
    
    # CPD for A: A = beta_A (deterministic)
    cpd_A = TabularCPD('A', 2, 
                       [[1, 0],   # A=0 when beta_A=0, A=1 when beta_A=1
                        [0, 1]], 
                       evidence=['beta_A'], evidence_card=[2])
    
    # CPD for B: B = (A AND theta_A_B) OR beta_B (deterministic)
    # B depends on A, theta_A_B, and beta_B
    cpd_B_values = []
    for beta_b in [0, 1]:
        for theta_ab in [0, 1]:
            for a in [0, 1]:
                # B = (A AND theta_A_B) OR beta_B
                causal_contrib = a and theta_ab
                b_value = int(causal_contrib or beta_b)
                cpd_B_values.append([1-b_value, b_value])
    
    cpd_B = TabularCPD('B', 2, 
                       list(map(list, zip(*cpd_B_values))),
                       evidence=['beta_B', 'theta_A_B', 'A'], 
                       evidence_card=[2, 2, 2])
    
    # CPD for C: C = (A AND theta_A_C) OR beta_C (deterministic)
    cpd_C_values = []
    for beta_c in [0, 1]:
        for theta_ac in [0, 1]:
            for a in [0, 1]:
                # C = (A AND theta_A_C) OR beta_C
                causal_contrib = a and theta_ac
                c_value = int(causal_contrib or beta_c)
                cpd_C_values.append([1-c_value, c_value])
    
    cpd_C = TabularCPD('C', 2, 
                       list(map(list, zip(*cpd_C_values))),
                       evidence=['beta_C', 'theta_A_C', 'A'], 
                       evidence_card=[2, 2, 2])
    
    # CPD for D: D depends on B, C, theta_B_D, theta_C_D, and beta_D
    cpd_D_values = []
    for beta_d in [0, 1]:
        for theta_cd in [0, 1]:
            for theta_bd in [0, 1]:
                for c in [0, 1]:
                    for b in [0, 1]:
                        # Causal contributions - now using separate powers
                        b_contrib = b and theta_bd
                        c_contrib = c and theta_cd
                        
                        if disjunction:
                            # D = (B_contrib OR C_contrib) OR beta_D
                            causal_contrib = b_contrib or c_contrib
                        else:
                            # D = (B_contrib AND C_contrib) OR beta_D
                            causal_contrib = b_contrib and c_contrib
                        
                        d_value = int(causal_contrib or beta_d)
                        cpd_D_values.append([1-d_value, d_value])
    
    cpd_D = TabularCPD('D', 2, 
                       list(map(list, zip(*cpd_D_values))),
                       evidence=['beta_D', 'theta_C_D', 'theta_B_D', 'C', 'B'], 
                       evidence_card=[2, 2, 2, 2, 2])
    
    # Add CPDs to the model
    model.add_cpds(cpd_beta_A, cpd_beta_B, cpd_beta_C, cpd_beta_D,
                   cpd_theta_A_B, cpd_theta_A_C, cpd_theta_B_D, cpd_theta_C_D,
                   cpd_A, cpd_B, cpd_C, cpd_D)
    
    # Verify the model and return its parameters if valid
    if model.check_model():
        return model
    else:
        # Return the parameters that caused the failure
        failed_params = {
            'power_B': power_B,
            'power_C': power_C,
            'power_D_from_B': power_D_from_B,
            'power_D_from_C': power_D_from_C,
            'beta_A': beta_A,
            'beta_B': beta_B,
            'beta_C': beta_C,
            'beta_D': beta_D,
            'disjunction': disjunction
        }
        raise ValueError(f"Model failed validation checks with parameters: {failed_params}")
