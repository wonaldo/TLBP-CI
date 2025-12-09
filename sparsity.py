import transfer
import importlib
import data_generator as dg
importlib.reload(transfer)
importlib.reload(dg)
import HSICtest
importlib.reload(HSICtest)
import graph_generator as gg
from algorithm import hsic_te
from metrics import adj_matrix, SHD_distance, count_accuracy, compute_f1




## Varying degrees of sparsity
graph_1=gg.renyi_graph(10, 0.1,seed=49)
graph_3=gg.renyi_graph(10, 0.3,seed=49)
graph_5=gg.renyi_graph(10, 0.5,seed=49)


if __name__ == '__main__':
    res_list=[]
    sta_list=[]
    shd_list=[]
    fdr_list=[]
    f1_list=[]
    for m in [graph_1,graph_3,graph_5]:
        num_samples = 1000
        data=dg.generate_data_with_causal_and_autocorrelation(m, num_samples, noise_std=0.4)
        nodes = list(set([u for u, v in m] + [v for u, v in m]))
        adjacency,ad=hsic_te(data,nodes,bandwidth=2.0,alpha=0.05)
        res_list.append(ad)
        sta_list.append(adjacency)
    for i in range(len(res_list)):
        adj_sta=sta_list[i]
        adj_te=res_list[i]       
        shd = SHD_distance(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
        fdr = count_accuracy(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
        f1= compute_f1(adj_matrix(10, adj_sta), adj_matrix(10, adj_te))
        shd_list.append(shd)
        fdr_list.append(fdr)
        f1_list.append(f1)
    print("shd-distance:",shd_list)
    print("fdr:",fdr_list)
    print("f1-score:",f1_list)