import numpy as np
import transfer
import importlib

import data_generator as dg
import matplotlib.pyplot as plt
import HSICtest
from algorithm import hsic_te

from metrics import adj_matrix, SHD_distance, count_accuracy, compute_f1
importlib.reload(transfer)
importlib.reload(dg)
importlib.reload(HSICtest)



## Inspection of three structures
## Confounders
edges1 = [
    (0, 1),
    (0, 2),

    (3, 4),
    (3, 5),


    (6, 7),
    (6, 8),


    (2, 9),
    (2, 5),

    (1, 3),
    (4, 6)
]

## Colliders
edges2 = [

    (0, 2),
    (1, 2),

    (3, 4),
    (5, 4),

    (6, 7),
    (8, 7),

    (2, 9),
    (4, 9),
     (9, 6)
]

## Mediators
edges3 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),

    (0, 3),  
    (2, 5),   
    (4, 7)    
]

# Hybrids
edges4 = [
    (2, 0), (2, 1),
    (6, 3), (6, 4), 

    (7, 8), (8, 9),
    (4, 5), (5, 3),

    (0, 5), (1, 5), 

    (5, 8),            
]


def edges_to_children_dict(edges):
    nodes = set()
    for p, c in edges:
        nodes.add(p)
        nodes.add(c)

    children_dict = {node: [] for node in nodes}

    for parent, child in edges:
        children_dict[parent].append(child)

    return children_dict


if __name__ == '__main__':
    ## Different Cases
    res_list=[]
    sta_list=[]
    shd_list=[]
    fdr_list=[]
    f1_list=[]
    for m in [edges1,edges2,edges3,edges4]:
        num_samples = 1000
        data=dg.generate_data_with_causal_and_autocorrelation(m, num_samples, noise_std=0.4)
        nodes = list(set([u for u, v in m] + [v for u, v in m]))
        adjacency,ad=hsic_te(data,nodes,bandwidth=2.0,alpha=0.05)
        res_list.append(ad)
        sta=edges_to_children_dict(m)
        sta_list.append(sta)
    print(res_list)

    ## Evaluation
    for i in range(len(res_list)):
       adj_sta=sta_list[i]
       adj_te=res_list[i]
       shd = SHD_distance(adj_matrix(10, adj_sta), adj_matrix(10, ad))
       fdr= count_accuracy(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
       f1 = compute_f1(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
       shd_list.append(shd)
       fdr_list.append(fdr)
       f1_list.append(f1)
    print("shd-distance:",shd_list)
    print("fdr:",fdr_list)
    print("f1-score:",f1_list)
