import numpy as np
import transfer
import importlib

import numpy as np
import matplotlib.pyplot as plt
import data_generator as dg
import graph_generator as gg
import matplotlib.pyplot as plt
importlib.reload(transfer)
importlib.reload(dg)
import HSICtest
importlib.reload(HSICtest)
from algorithm import hsic_te
from metrics import adj_matrix, SHD_distance, count_accuracy, compute_f1




# Different number of nodes
graph5_s=gg.renyi_graph(5, 0.1,seed=49)
graph10_s=gg.renyi_graph(10, 0.1,seed=49)
graph20_s=gg.renyi_graph(20, 0.1,seed=49)
graph30_s=gg.renyi_graph(30, 0.1, seed=49)
graph50_s=gg.renyi_graph(50, 0.1,seed=49)


if __name__ == '__main__':
    res_list=[]
    sta_list=[]
    for m in [graph5_s,graph10_s,graph20_s,graph30_s,graph50_s]:
        num_samples = 1000
        data=dg.generate_data_with_causal_and_autocorrelation(m, num_samples, noise_std=0.4)
        nodes = list(set([u for u, v in m] + [v for u, v in m]))
        nnodes=len(nodes)
        adjacency,ad=hsic_te(data,nodes,bandwidth=2.0,alpha=0.05)
        res_list.append(ad)
        sta_list.append(adjacency)
    print(res_list)
 
    ## nnodes=5
    adj_sta=sta_list[0]
    adj_te=res_list[0]
    shd1 = SHD_distance(adj_matrix(5, adj_sta), adj_matrix(5, adj_te))
    fdr1 = count_accuracy(adj_matrix(5, adj_sta), adj_matrix(5, adj_te))
    f11 = compute_f1(adj_matrix(5, adj_sta), adj_matrix(5, adj_te))
    print("nnodes_5:",shd1,fdr1,f11)

    ## nnodes=10
    adj_sta=sta_list[1]
    adj_te=res_list[1]
    shd2 = SHD_distance(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
    fdr2 = count_accuracy(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
    f12 = compute_f1(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
    print("nnodes_10:",shd2,fdr2,f12)

    ## nnodes=20
    adj_sta=sta_list[2]
    adj_te=res_list[2]
    shd3 = SHD_distance(adj_matrix(20, adj_sta), adj_matrix(20, adj_te))
    fdr3 = count_accuracy(adj_matrix(20, adj_sta), adj_matrix(20, adj_te))
    f13 = compute_f1(adj_matrix(20, adj_sta), adj_matrix(20, adj_te))
    print("nnodes_20:",shd3,fdr3,f13)

    ## nnodes=30
    adj_sta=sta_list[3]
    adj_te=res_list[3]
    shd4 = SHD_distance(adj_matrix(30, adj_sta), adj_matrix(30, adj_te))
    fdr4 = count_accuracy(adj_matrix(30, adj_sta), adj_matrix(30, adj_te))
    f14 = compute_f1(adj_matrix(30, adj_sta), adj_matrix(30, adj_te))
    print("nnodes_30:",shd4,fdr4,f14)

    ## nnodes=50
    adj_sta=sta_list[4]
    adj_te=res_list[4]
    shd5 = SHD_distance(adj_matrix(50, adj_sta), adj_matrix(50, adj_te))
    fdr5 = count_accuracy(adj_matrix(50, adj_sta), adj_matrix(50, adj_te))
    f15 = compute_f1(adj_matrix(50, adj_sta), adj_matrix(50, adj_te))
    print("nnodes_50:",shd5,fdr5,f15)
