import numpy as np
import networkx as nx

def generate_data_with_causal_and_autocorrelation(edges, num_samples, noise_std=0.4,seed=49,lag=1):
    """
    :param edges: A list of edges, representing the structure of the causal graph
    :param num_samples: Number of samples
    :param noise_std: Standard deviation of noise
    :return: Synthetic data
    """
    np.random.seed(seed)
    ## Construct the DAGs
    G = nx.DiGraph()
    G.add_edges_from(edges)

    num_nodes = len(G.nodes)
    data = np.zeros((num_samples, num_nodes))
    #### Different Noise Types: Note that only one type of noise code is kept in a non-commented state. 

    ### Gaussian Noise
    noise = np.random.normal(0, noise_std, size=(num_samples, num_nodes))
    
    # ### Laplace Noise
    # b = noise_std / np.sqrt(2)
    # noise = np.random.laplace(0, b, size=(num_samples, num_nodes))
    
    # ## Uniform Noise
    # a = np.sqrt(3) * noise_std
    # noise = np.random.uniform(-a, a, size=(num_samples, num_nodes))

    ## Beginning X0 ~ N(0, Id)
    for k in range(0,lag):
        data[k, :] = np.random.normal(0, 1, num_nodes)

    if lag==1:
        for tau in range(1, num_samples):
            for i in G.nodes:
                ### Get the parent of the current node (according to the causal graph)
                parents = list(G.predecessors(i))  # Get all parents pointing to the current node
                
                ### Calculate current node data with causality and autocorrelation
                X_prev = data[tau - 1, list(G.nodes).index(i)]  ## X_0
                
                ## Autocorrelation.
                ## Linear Causality
                # term0 = 0.6*X_prev  # Linear Equation 1  
                # term0 = 0.5*X_prev  # Linear Equation 2
                
                ## Non-Linear Causality
                term0 = 5/(1+np.exp(-X_prev)) # Non-Linear Equation 1  
                # term0 = np.sin(X_prev) # Non-Linear Equation 2
                

                ## Causal effects of parent nodes
                if parents:
                    term1=0
                    for parent in parents:
                        ## Linear Causality
                        # term1 += 0.2 * data[tau-1, list(G.nodes).index(parent)]  # Linear Equation 1 
                        # term1 += 0.4 * data[tau-1, list(G.nodes).index(parent)]  # Linear Equation 2

                        ## Non-Linear Causality
                        term1 += 1/(1+np.exp(-data[tau-1, list(G.nodes).index(parent)]))  # Non-Linear Equation 1
                        # term1 += 0.2*np.sin(data[tau-1, list(G.nodes).index(parent)])     # Non-Linear Equation 2
                    
                    noise_term = noise[tau, list(G.nodes).index(i)]  # Noise Term
                    data[tau, list(G.nodes).index(i)] = term0 + term1 + noise_term
                else:
                    noise_term = noise[tau, list(G.nodes).index(i)]  # Noise Term
                    data[tau, list(G.nodes).index(i)] = term0 + noise_term
    else:  ## Time-lag>1
        for tau in range(lag, num_samples):
            for i in G.nodes:
                ### Get the parent of the current node (according to the causal graph)
                parents = list(G.predecessors(i))  # Get all parents pointing to the current node
                
                ### Calculate current node data with causality and autocorrelation
                X_prev = data[tau - lag, list(G.nodes).index(i)]  ## X_0
                
                ## Autocorrelation.
                ## Linear Causality
                term0 = 0.6*X_prev  # Linear Equation 1  
                # term0 = 0.5*X_prev  # Linear Equation 2
                
                ## Non-Linear Causality
                # term0 = 5/(1+np.exp(-X_prev)) # Non-Linear Equation 1  
                # term0 = np.sin(X_prev) # Non-Linear Equation 2
                

                ## Causal effects of parent nodes
                if parents:
                    term1=0
                    for parent in parents:
                        ## Linear Causality
                        term1 += 0.2 * data[tau-lag, list(G.nodes).index(parent)]  # Linear Equation 1 
                        # term1 += 0.4 * data[tau-1, list(G.nodes).index(parent)]  # Linear Equation 2

                        ## Non-Linear Causality
                        # term1 += 1/(1+np.exp(-data[tau-1, list(G.nodes).index(parent)]))  # Non-Linear Equation 1
                        # term1 += 0.2*np.sin(data[tau-1, list(G.nodes).index(parent)])     # Non-Linear Equation 2
                    
                    noise_term = noise[tau, list(G.nodes).index(i)]  # Noise Term
                    data[tau, list(G.nodes).index(i)] = term0 + term1 + noise_term
                else:
                    noise_term = noise[tau, list(G.nodes).index(i)]  # Noise Term
                    data[tau, list(G.nodes).index(i)] = term0 + noise_term 
    return data