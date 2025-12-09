import numpy as np
import transfer
import importlib
import data_generator as dg
import HSICtest
from algorithm import hsic_te
from metrics import adj_matrix, SHD_distance, count_accuracy, compute_f1
importlib.reload(transfer)
importlib.reload(dg)
importlib.reload(HSICtest)


# Identification of latent factors
edges=[
    (0,1),
    (0,2),
    (1,2),
    (1,3),
    (1,4),
    (2,5),
    (2,6),
    (6,7)
]

if __name__ == '__main__':
    num_samples = 1000
    ## Control linear/non-linear by changing equations in data_generator.py
    data=dg.generate_data_with_causal_and_autocorrelation(edges, num_samples, noise_std=0.4)
    keep_cols=[0,3,4,5,6,7]
    data = data[:, keep_cols]
    nodes=keep_cols
    adjacency,ad=hsic_te(data,nodes,bandwidth=2.0,alpha=0.05)
    shd = SHD_distance(adj_matrix(10, adjacency), adj_matrix(10, ad))
    fdr = count_accuracy(adj_matrix(10, adjacency), adj_matrix(10, ad))
    f1 = compute_f1(adj_matrix(10, adjacency), adj_matrix(10, ad))