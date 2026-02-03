from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

class CounterfactualInferenceBN:
    def __init__(self, base_model):
        """
        Initializes the CounterfactualInferenceBN class with a base Bayesian Network model.
        
        Args:
            base_model (DiscreteBayesianNetwork): The base Bayesian Network model
            base_inference (VariableElimination): The inference engine for the base model
            s (float): Stability parameter controlling influence of real exogenous values on counterfactual ones
        """
        

        self.base_model = base_model
        self.base_inference = VariableElimination(self.base_model)

        # Identify exogenous and endogenous variables from base model
        self._identify_variable_types()

        self._stability = None
        self._twin_inference = None
        self._chain_inference = None
        
    def _identify_variable_types(self):
        """Identify exogenous and endogenous variables from the base model structure."""
        all_nodes = set(self.base_model.nodes())
        
        # Exogenous variables are those that start with 'beta_' or 'theta_'
        self.exogenous_vars = {node for node in all_nodes 
                              if node.startswith('beta_') or node.startswith('theta_')}
        
        # Endogenous variables are the remaining nodes
        self.endogenous_vars = all_nodes - self.exogenous_vars

    def _normalize_stability_param(self, s):
        """
        Normalize stability parameter to a dictionary format.
        
        Args:
            s (float, dict, or None): Stability parameter(s). Can be:
                - float: Same stability for all exogenous variables
                - dict: Variable-specific stability {var_name: stability_value}
                - None: Defaults to 1.0 for all variables
        
        Returns:
            dict: Dictionary mapping each exogenous variable to its stability value
        """
        if s is None:
            s = 1.0
            
        if isinstance(s, (int, float)):
            # Single stability value for all exogenous variables
            return {var: float(s) for var in self.exogenous_vars}
        elif isinstance(s, dict):
            # Variable-specific stability values
            stability_dict = {}
            for var in self.exogenous_vars:
                if var in s:
                    stability_dict[var] = float(s[var])
                else:
                    # Default to 1.0 for variables not specified
                    stability_dict[var] = 1.0
            return stability_dict
        else:
            raise ValueError("Stability parameter must be float, dict, or None")

    def _create_twin_network(self, stability_dict):
        """
        Creates a twin network with both real and counterfactual variables.
        The twin network includes:
        1. All original variables (real world)
        2. Counterfactual versions of endogenous variables
        3. Counterfactual versions of exogenous variables
        4. Stability connections from real to counterfactual exogenous variables
        
        Args:
            stability_dict (dict): Dictionary mapping each exogenous variable to its stability value
        """
        
            
        # Create new model for twin network
        twin_model = DiscreteBayesianNetwork()
        
        # Add all original nodes
        for node in self.base_model.nodes():
            twin_model.add_node(node)
        
        # Add counterfactual versions of all variables
        cf_nodes = []
        for node in self.base_model.nodes():
            cf_node = f"{node}_cf"
            cf_nodes.append(cf_node)
            twin_model.add_node(cf_node)
        
        # Add all original edges
        for edge in self.base_model.edges():
            twin_model.add_edge(edge[0], edge[1])
        
        # Add counterfactual edges (mirror structure for cf variables)
        for edge in self.base_model.edges():
            cf_edge = (f"{edge[0]}_cf", f"{edge[1]}_cf")
            twin_model.add_edge(cf_edge[0], cf_edge[1])
        
        # Add stability edges from real to counterfactual exogenous variables
        for exo_var in self.exogenous_vars:
            twin_model.add_edge(exo_var, f"{exo_var}_cf")
        
        # Copy CPDs for original variables
        cpds_to_add = []
        for cpd in self.base_model.get_cpds():
            cpds_to_add.append(cpd)
        
        # Create CPDs for counterfactual endogenous variables (same structure as originals)
        for cpd in self.base_model.get_cpds():
            if cpd.variable in self.endogenous_vars:
                # Create counterfactual version of this CPD
                cf_variable = f"{cpd.variable}_cf"
                cf_evidence = [f"{var}_cf" for var in cpd.variables[1:]] if len(cpd.variables) > 1 else None
                cf_evidence_card = cpd.cardinality[1:] if len(cpd.cardinality) > 1 else None
                
                # Convert values to proper format
                if hasattr(cpd.values, 'numpy'):
                    cpd_array = cpd.values.numpy()
                elif hasattr(cpd.values, 'shape'):
                    cpd_array = np.array(cpd.values)
                else:
                    cpd_array = np.array(cpd.values)
                
                # Get variable cardinality (number of states for the variable)
                variable_card = cpd.cardinality[0]
                
                # Calculate total number of parent combinations
                if len(cpd.cardinality) > 1:
                    num_parent_combinations = np.prod(cpd.cardinality[1:])
                else:
                    num_parent_combinations = 1
                
                # Reshape to 2D: (variable_card, num_parent_combinations)
                if cpd_array.size == variable_card * num_parent_combinations:
                    cpd_array_2d = cpd_array.reshape(variable_card, num_parent_combinations)
                else:
                    if cpd_array.ndim == 1:
                        cpd_array_2d = cpd_array.reshape(-1, 1)
                    else:
                        cpd_array_2d = cpd_array.reshape(cpd_array.shape[0], -1)
                
                # Convert to list for TabularCPD
                cpd_values = cpd_array_2d.tolist()
                
                cf_cpd = TabularCPD(
                    variable=cf_variable,
                    variable_card=variable_card,
                    values=cpd_values,
                    evidence=cf_evidence,
                    evidence_card=cf_evidence_card
                )
                cpds_to_add.append(cf_cpd)
        
        # Create stability CPDs for counterfactual exogenous variables
        for exo_var in self.exogenous_vars:
            cf_exo_var = f"{exo_var}_cf"

            base_exo_var = exo_var.replace('_cf', '')
            # Get original prior probability from base model
            orig_cpd = self.base_model.get_cpds(base_exo_var)

            # # Handle values conversion for original CPD too
            # if hasattr(orig_cpd.values, 'numpy'):
            #     orig_array = orig_cpd.values.numpy()
            # elif hasattr(orig_cpd.values, 'shape'):
            #     orig_array = np.array(orig_cpd.values)
            # else:
            #     orig_array = np.array(orig_cpd.values)
                
            # # Ensure correct shape for accessing - flatten if necessary
            # if orig_array.ndim == 1:
            #     orig_array = orig_array.reshape(-1, 1)
            # elif orig_array.ndim > 2:
            #     orig_array = orig_array.reshape(orig_array.shape[0], -1)

            prior_prob = orig_cpd.values[1]  # P(U=1)

            # Stability CPD values using the provided s parameter
            s_var = stability_dict[exo_var]
            prob_cf1_given_u0 = (1 - s_var) * prior_prob
            prob_cf1_given_u1 = s_var + (1 - s_var) * prior_prob
            
            stability_values = [
                [1 - prob_cf1_given_u0, 1 - prob_cf1_given_u1],  # P(U_cf=0 | U-0), P(U_cf = 0 | U=1)
                [prob_cf1_given_u0, prob_cf1_given_u1]   # P(U_cf=1 | U = 0), P(U_cf = 1 | U=1)
            ]
            
            stability_cpd = TabularCPD(
                variable=cf_exo_var,
                variable_card=2,
                values=stability_values,
                evidence=[exo_var],
                evidence_card=[2]
            )
            cpds_to_add.append(stability_cpd)
        
        # Add all CPDs to the twin model
        twin_model.add_cpds(*cpds_to_add)

        # Verify the model
        assert twin_model.check_model(), "Twin network model is invalid"
        
        return twin_model

    def _create_bp_twin(self, cf_query=None):
        """
        Create a twin model that applies Pearl-style do() interventions to any
        variables listed in cf_query. This is intended to be used when s == 1.
        - For each var in cf_query: remove incoming edges to var_cf and set a
          deterministic CPD do(var_cf = value).
        - Otherwise mirror the base model (original + _cf copies) and add stability
          CPDs for exogenous variables as in _create_twin_network.
        """
        

        twin_model = DiscreteBayesianNetwork()

        # Add original nodes and cf nodes
        for node in self.base_model.nodes():
            twin_model.add_node(node)
            twin_model.add_node(f"{node}_cf")

        # Add original edges and mirror cf structure
        for u, v in self.base_model.edges():
            twin_model.add_edge(u, v)
            twin_model.add_edge(f"{u}_cf", f"{v}_cf")

        # Add stability edges from real exogenous -> cf exogenous
        for exo_var in self.exogenous_vars:
            twin_model.add_edge(exo_var, f"{exo_var}_cf")

        # Collect CPDs to add (copy originals)
        cpds_to_add = []
        for cpd in self.base_model.get_cpds():
            cpds_to_add.append(cpd)

        # Create CPDs for counterfactual endogenous variables,
        # but skip those that will be intervened by cf_query (they will get deterministic CPDs)
        cf_targets = set()
        if cf_query:
            for var in cf_query:
                clean_var = var.replace("_cf", "")
                cf_targets.add(f"{clean_var}_cf")

        for cpd in self.base_model.get_cpds():
            if cpd.variable in self.endogenous_vars:
                cf_variable = f"{cpd.variable}_cf"
                if cf_variable in cf_targets:
                    # Skip creating structural cf-CPD for intervened variable
                    continue

                cf_evidence = [f"{var}_cf" for var in cpd.variables[1:]] if len(cpd.variables) > 1 else None
                cf_evidence_card = cpd.cardinality[1:] if len(cpd.cardinality) > 1 else None

                # Normalize and reshape original values to (var_card, parent_combinations)
                if hasattr(cpd.values, "numpy"):
                    cpd_array = cpd.values.numpy()
                elif hasattr(cpd.values, "shape"):
                    cpd_array = np.array(cpd.values)
                else:
                    cpd_array = np.array(cpd.values)

                variable_card = cpd.cardinality[0]
                if len(cpd.cardinality) > 1:
                    num_parent_combinations = np.prod(cpd.cardinality[1:])
                else:
                    num_parent_combinations = 1

                if cpd_array.size == variable_card * num_parent_combinations:
                    cpd_array_2d = cpd_array.reshape(variable_card, num_parent_combinations)
                else:
                    if cpd_array.ndim == 1:
                        cpd_array_2d = cpd_array.reshape(-1, 1)
                    else:
                        cpd_array_2d = cpd_array.reshape(cpd_array.shape[0], -1)

                cpd_values = cpd_array_2d.tolist()

                cf_cpd = TabularCPD(
                    variable=cf_variable,
                    variable_card=variable_card,
                    values=cpd_values,
                    evidence=cf_evidence,
                    evidence_card=cf_evidence_card,
                )
                cpds_to_add.append(cf_cpd)

        # Create stability CPDs for counterfactual exogenous variables
        for exo_var in self.exogenous_vars:
            cf_exo_var = f"{exo_var}_cf"
            base_exo_var = exo_var.replace("_cf", "")
            orig_cpd = self.base_model.get_cpds(base_exo_var)
            prior_prob = orig_cpd.values[1]  # P(U=1)

            prob_cf1_given_u0 = (1 - 1.0) * prior_prob  # s==1 handled elsewhere; keep formula consistent
            prob_cf1_given_u1 = 1.0 + (1 - 1.0) * prior_prob

            # If this cf exogenous is targeted by intervention, we will remove stability edge and set deterministic CPD later.
            if cf_exo_var in cf_targets:
                # do not add stability CPD for intervened exogenous variable
                continue

            stability_values = [
                [1 - prob_cf1_given_u0, 1 - prob_cf1_given_u1],
                [prob_cf1_given_u0, prob_cf1_given_u1],
            ]

            stability_cpd = TabularCPD(
                variable=cf_exo_var, variable_card=2, values=stability_values, evidence=[exo_var], evidence_card=[2]
            )
            cpds_to_add.append(stability_cpd)

        # Add non-intervened CPDs to model
        twin_model.add_cpds(*cpds_to_add)

        # Now apply do() interventions: for each cf target, remove incoming edges and set deterministic CPD
        if cf_query:
            for var, val in cf_query.items():
                clean_var = var.replace("_cf", "")
                cf_var = f"{clean_var}_cf"

                # Remove incoming edges to cf_var (parents -> cf_var)
                # collect parents to avoid modifying iterator
                incoming_parents = [p for p, c in list(twin_model.edges()) if c == cf_var]
                for p in incoming_parents:
                    try:
                        twin_model.remove_edge(p, cf_var)
                    except Exception:
                        pass

                # Remove any existing CPD for cf_var
                try:
                    existing = [cpd for cpd in twin_model.get_cpds() if getattr(cpd, "variable", None) == cf_var]
                    if existing:
                        twin_model.remove_cpds(*existing)
                except Exception:
                    pass

                # Add deterministic intervention CPD do(cf_var = val)
                # values should be listed per-state: [P(state=0)], [P(state=1)]
                intervention_cpd = TabularCPD(variable=cf_var, variable_card=2, values=[[1 - val], [val]])
                twin_model.add_cpds(intervention_cpd)

        # Final model validation
        assert twin_model.check_model(), "BP twin network model is invalid"
        return twin_model

    def _create_chained_twin_network(self, num_levels, s):
        """
        Creates a single large network with chained counterfactual variables.
        
        Args:
            num_levels (int): The number of counterfactual levels to create.
            s (float): The stability parameter.
            
        Returns:
            DiscreteBayesianNetwork: The complete chained twin network model.
        """
        chained_model = self.base_model.copy()
        cpds_to_add = []

        for level in range(num_levels):
            prev_suffix = '_cf' * level
            curr_suffix = '_cf' * (level + 1)

            # Add counterfactual nodes and structural edges for the current level
            for node in self.base_model.nodes():
                chained_model.add_node(f"{node}{curr_suffix}")
            for edge in self.base_model.edges():
                chained_model.add_edge(f"{edge[0]}{curr_suffix}", f"{edge[1]}{curr_suffix}")

            # Add stability edges from previous level's exogenous vars to current level's
            for exo_var in self.exogenous_vars:
                chained_model.add_edge(f"{exo_var}{prev_suffix}", f"{exo_var}{curr_suffix}")

            # Create and add CPDs for the new counterfactual variables
            for cpd in self.base_model.get_cpds():
                if cpd.variable in self.endogenous_vars:
                    # Create counterfactual version of this endogenous CPD
                    cf_variable = f"{cpd.variable}{curr_suffix}"
                    cf_evidence = [f"{var}{curr_suffix}" for var in cpd.variables[1:]] if len(cpd.variables) > 1 else None
                    
                    # The CPD values are identical to the original, but need reshaping
                    if hasattr(cpd.values, 'numpy'):
                        cpd_array = cpd.values.numpy()
                    else:
                        cpd_array = np.array(cpd.values)

                    variable_card = cpd.get_cardinality([cpd.variable])[cpd.variable]
                    
                    if len(cpd.variables) > 1:
                        num_parent_combinations = np.prod([cpd.get_cardinality([var])[var] for var in cpd.variables[1:]])
                    else:
                        num_parent_combinations = 1

                    if cpd_array.size == variable_card * num_parent_combinations:
                         cpd_array_2d = cpd_array.reshape(variable_card, num_parent_combinations)
                    else:
                        if cpd_array.ndim == 1:
                            cpd_array_2d = cpd_array.reshape(-1, 1)
                        else:
                            cpd_array_2d = cpd_array.reshape(cpd_array.shape[0], -1)

                    cpd_values = cpd_array_2d.tolist()

                    cf_cpd = TabularCPD(
                        variable=cf_variable,
                        variable_card=variable_card,
                        values=cpd_values,
                        evidence=cf_evidence,
                        evidence_card=[cpd.get_cardinality([var])[var] for var in cpd.variables[1:]] if cf_evidence else None
                    )
                    cpds_to_add.append(cf_cpd)

            # Create stability CPDs for counterfactual exogenous variables
            for exo_var in self.exogenous_vars:
                cf_exo_var = f"{exo_var}{curr_suffix}"
                prev_exo_var = f"{exo_var}{prev_suffix}"

                base_exo_var = exo_var.replace('_cf', '')

                orig_cpd = self.base_model.get_cpds(base_exo_var)

                prior_prob = orig_cpd.values[1]  # P(U=1)
                
                prob_cf1_given_u0 = (1 - s) * prior_prob
                prob_cf1_given_u1 = s + (1 - s) * prior_prob                


                stability_values = [
                    [1 - prob_cf1_given_u0, 1 - prob_cf1_given_u1],  # P(U_cf=0 | U-0), P(U_cf = 0 | U=1)
                    [prob_cf1_given_u0, prob_cf1_given_u1]   # P(U_cf=1 | U = 0), P(U_cf = 1 | U=1)
                ]

                stability_cpd = TabularCPD(
                    variable=cf_exo_var,
                    variable_card=2,
                    values=stability_values,
                    evidence=[prev_exo_var],
                    evidence_card=[2]
                )
                cpds_to_add.append(stability_cpd)

        chained_model.add_cpds(*cpds_to_add)
        assert chained_model.check_model(), "Chained twin network model is invalid"
        return chained_model
    
    def _get_twin_inference_engine(self, s, cf_query=None):
        """Get or create inference engine for the twin network."""

        stability_dict = self._normalize_stability_param(s)
        
        if s == 1 and cf_query:
            twin_model = self._create_bp_twin(cf_query)
        else:
            twin_model = self._create_twin_network(stability_dict)
        
        twin_inference = VariableElimination(twin_model)
        
        return twin_inference
    
    def query(self, variables, evidence=None, cf_evidence=None, marginals=False, s=None, verbose=False):
        """
        Perform inference using the twin network, similar to VariableElimination.query().
        
        Args:
            variables (list): List of variables to query (can include exogenous or endogenous variables)
            evidence (dict, optional): Observations of real variables
            cf_evidence (dict, optional): Observations of counterfactual variables  
            marginals (bool): If True, return individual marginal probabilities instead of joint distribution
            s (float, optional): Stability parameter. If None, defaults to 1
            verbose (bool): If True, print which inference engine is being used
        
        Returns:
            dict or pgmpy Factor: If marginals=True, returns dict mapping each variable to P(var=1|evidence).
                                 Otherwise returns joint distribution (dict for all queries)
        """
        # Handle default arguments
        if evidence is None:
            evidence = {}
        if cf_evidence is None:
            cf_evidence = {}
            
        # Always use twin network
        
        if self._twin_inference is None:
            self._stability = s
            self._twin_inference = self._get_twin_inference_engine(s, cf_query=cf_evidence)
        if s != self._stability or s == 1:
            self._stability = s
            self._twin_inference = self._get_twin_inference_engine(s, cf_query=cf_evidence)
        twin_inference = self._twin_inference
        
        # Prepare evidence for twin network
        all_evidence = evidence.copy()
        
        # Add counterfactual observations with proper naming
        for var, val in cf_evidence.items():
            clean_var = var.replace('_cf', '')
            cf_var = f"{clean_var}_cf"
            all_evidence[cf_var] = val
        
        if verbose:
            print(f"Evidence for twin network: {all_evidence}")

        if marginals:
            # Return marginal probabilities for requested variables
            marginal_probs = {}
            
            for var in variables:
                # Determine which version of variable to query
                query_var = f"{var}_cf"
                    
                marginal = twin_inference.query(variables=[query_var], evidence=all_evidence)
                marginal_probs[var] = marginal.values[1]  # P(var=1|evidence)
            
            return marginal_probs
        else:
            # Return joint distribution over requested variables
            if verbose:
                print(f"Querying joint distribution for variables: {variables}")
            query_vars = []
            for var in variables:
                query_vars.append(f"{var}_cf")
                
            joint_dist = twin_inference.query(variables=query_vars, evidence=all_evidence)
            
            if verbose:
                # print the joint distribution factor
                print(f"Joint distribution factor:\n{joint_dist}")

            return joint_dist
            # Always convert to dict format for consistency
            # result = {}
            # all_combinations = list(product([0, 1], repeat=len(query_vars)))

    
            # for i, combination in enumerate(all_combinations):
            #     config_items = []
            #     for j, query_var in enumerate(query_vars):
            #         clean_var = query_var.replace('_cf', '')  # Remove _cf suffix for consistency
            #         config_items.append((clean_var, combination[j]))
                
            #     config_tuple = tuple(config_items)
                
            #     # Create index tuple for accessing joint_dist.values
            #     index_tuple = tuple(combination)
            #     prob = joint_dist.values[index_tuple]
                
            #     if prob > 1e-10:  # Only store non-negligible probabilities
            #         result[config_tuple] = prob

            # if verbose:
            #     print(f"Joint distribution result (dict format): {result}")
            
            # return result


    def query2(self, variables, evidence=None, cf_evidence=None, marginals=False, s=None, verbose=False):
        """
        Perform inference using twin networks, supporting chained counterfactual reasoning.
        
        Args:
            variables (list): List of variables to query (can include exogenous or endogenous variables)
            evidence (dict, optional): Observations of real variables
            cf_evidence (dict or list, optional): Counterfactual evidence. Can be:
                - dict: Single counterfactual scenario (creates one twin network)
                - list of dicts: Multiple chained counterfactual scenarios (creates chain of twin networks)
            marginals (bool): If True, return individual marginal probabilities instead of joint distribution
            s (float, optional): Stability parameter. If None, defaults to 1
            verbose (bool): If True, print which inference engine is being used
        
        Returns:
            dict or pgmpy Factor: If marginals=True, returns dict mapping each variable to P(var=1|evidence).
                                Otherwise returns joint distribution
        """
        # Handle default arguments
        if evidence is None:
            evidence = {}
        if cf_evidence is None:
            cf_evidence = {}
        
        # Convert single dict to list for uniform handling
        if isinstance(cf_evidence, dict):
            cf_evidence_list = [cf_evidence] if cf_evidence else []
        else:
            cf_evidence_list = cf_evidence
        
        # If no counterfactual evidence, use base model
        if not cf_evidence_list:
            if verbose:
                print('Use twin network: False')
                print("Using base model inference engine")
            base_inference = self.base_inference
            
            if marginals:
                marginal_probs = {}
                for var in variables:
                    marginal = base_inference.query(variables=[var], evidence=evidence)
                    marginal_probs[var] = marginal.values[1]
                return marginal_probs
            else:
                return base_inference.query(variables=variables, evidence=evidence)
        
        # Create chained twin networks
        if verbose:
            print('Use twin network: True')
            print(f"Creating chain of {len(cf_evidence_list)} twin networks (s={s})")
        
        
        if self._chain_inference is None:
            chain_model = self._create_chained_twin_network(len(cf_evidence_list), s)
            self._chain_inference = VariableElimination(chain_model)
        if s != self._stability:
            self._stability = s
            chain_model = self._create_chained_twin_network(len(cf_evidence_list), s)
            self._chain_inference = VariableElimination(chain_model)
        final_inference = self._chain_inference


        # Prepare evidence for chained twin network
        all_evidence = evidence.copy()
        
        # Add counterfactual evidence from each level of the chain
        for level, cf_ev in enumerate(cf_evidence_list):
            suffix = '_cf' + ('_cf' * level)  # _cf, _cf_cf, _cf_cf_cf, etc.
            for var, val in cf_ev.items():
                clean_var = var.replace('_cf', '')
                cf_var = f"{clean_var}{suffix}"
                all_evidence[cf_var] = val
        
        if verbose:
            print(f"Evidence for chained twin network: {all_evidence}")
        
        # Determine which variables to query (from the final level)
        final_suffix = '_cf' + ('_cf' * (len(cf_evidence_list) - 1))
        query_vars = [f"{var}{final_suffix}" for var in variables]
        
        if marginals:
            # Return marginal probabilities for requested variables
            marginal_probs = {}
            
            for i, var in enumerate(variables):
                query_var = query_vars[i]
                marginal = final_inference.query(variables=[query_var], evidence=all_evidence)
                marginal_probs[var] = marginal.values[1]  # P(var=1|evidence)
            
            return marginal_probs
        else:
            # Return joint distribution over requested variables
            if verbose:
                print(f"Querying joint distribution for variables: {query_vars}")
            
            joint_dist = final_inference.query(variables=query_vars, evidence=all_evidence)
            
            if verbose:
                print(f"Joint distribution factor:\n{joint_dist}")
            
            return joint_dist

