from causallearn.utils.KCI.KCI import *
import networkx as nx


CInd = KCI_CInd()
UInd = KCI_UInd()

def UIndtest(X, Y):
    return UInd.compute_pvalue(X,Y)

def CIndtest(X,Y,Z):
    return CInd.compute_pvalue(X,Y,Z)



def p4nodes(nodes,data,alpha):
    adjacency={node:[]for node in nodes}
    for i in range(len(nodes)):
        for j in range(len(nodes)):
                if i==j:
                    if UIndtest(data[1:,i].reshape(-1,1),data[1:,j].reshape(-1,1))[0]<alpha:
                        adjacency[nodes[i]].append(nodes[j])
                else:
                    if UIndtest(data[1:,i].reshape(-1,1),data[1:,j].reshape(-1,1))[0]<alpha:
                        adjacency[nodes[i]].append(nodes[j])
    return adjacency

def create_graph(adjacency):
    G=nx.DiGraph()
    G.add_edges_from(adjacency.keys())
    for u in adjacency:
        for v in adjacency[u]:
            G.add_edge(u,v)
    return G
