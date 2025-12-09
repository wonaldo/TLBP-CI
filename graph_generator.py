import networkx as nx

def renyi_graph(n,p,seed):
    g = nx.erdos_renyi_graph(n, p, seed=seed,directed=True)  # Directed Graph
    # Acyclicity Constraint
    dag = nx.DiGraph([(u, v) for u, v in g.edges() if u < v])
    edges= list(dag.edges())    
    return edges