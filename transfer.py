import itertools as itr
import kde
import importlib
import KCItest
importlib.reload(kde)
import numpy as np

def get_all_neighbors(G, nodes):
    adjacencies={}
    for node in nodes:
        in_neighbors = set(G.predecessors(node))
        out_neighbors = set(G.successors(node))
        adjacencies[node] = in_neighbors.union(out_neighbors)
    return adjacencies


def get_in_neighbours(G,nodes):
    in_adjacencies = {}
    for node in nodes:
        in_neighbors = set(G.predecessors(node))
        in_adjacencies[node] = in_neighbors
    return in_adjacencies

def get_out_neighbours(G, nodes):
    out_adjacencies = {}
    for node in nodes:
        out_neighbors = set(G.successors(node))
        out_adjacencies[node] = out_neighbors
    return out_adjacencies

def al4causal(nodes, data, bandwidth, adjacency,lag):
    new_ad = {node: [] for node in nodes}
    node_to_col_idx = {node: idx for idx, node in enumerate(nodes)}

    for i, j in itr.product(nodes, repeat=2):
        if j not in adjacency[i]:  # Ensure that there is an edge between i and j
            continue

        # Autocorrelation
        if i == j:
            sep_set_same = [node for node in adjacency[i] if node != i]
            # Mapping node numbers to column indices in data
            sep_set_same = [data[:, [node_to_col_idx[k]]] for k in sep_set_same if k in node_to_col_idx]
            sep_set_for_indtest = np.hstack(sep_set_same) if sep_set_same else np.empty((data.shape[0], 0))

            if sep_set_for_indtest.any():
                p_value = HSICtest.CIndtest(data[0:-1, [node_to_col_idx[i]]].reshape(-1, 1), data[1:, [node_to_col_idx[j]]].reshape(-1, 1), sep_set_for_indtest[0:-1, :])[0]
            else:
                p_value = HSICtest.UIndtest(data[0:-1, [node_to_col_idx[i]]].reshape(-1, 1), data[1:, [node_to_col_idx[j]]].reshape(-1, 1))[0]

            if p_value < 0.05:
                new_ad[i].append(j)

        # Causality from parents
        elif i != j:
            sep_set_i = [node for node in adjacency[i] if node != i]
            sep_set_j = [node for node in adjacency[j] if node != j]
            # Mapping node numbers to column indices in data
            sep_set_i = [data[:, [node_to_col_idx[k]]].reshape(-1, 1) for k in sep_set_i if k in node_to_col_idx]
            sep_set_j = [data[:, [node_to_col_idx[k]]].reshape(-1, 1) for k in sep_set_j if k in node_to_col_idx]


            ## Calculate the correlation first
            sep_set_for_indtest = np.hstack(sep_set_i) if sep_set_i else np.empty((data.shape[0], 0))
            ## print(sep_set_for_indtest.shape)
            ## 不同lag下的作用
            # if lag==1:
            #     if sep_set_for_indtest.any():
            #         p_value = HSICtest.CIndtest(data[2:, [node_to_col_idx[i]]].reshape(-1, 1), data[2:, [node_to_col_idx[j]]].reshape(-1, 1), sep_set_for_indtest[1:-1, :])[0]
            #     else:
            #         p_value = HSICtest.UIndtest(data[2:, [node_to_col_idx[i]]].reshape(-1, 1), data[2:, [node_to_col_idx[j]]].reshape(-1, 1))[0]
            # elif lag>1:
            #     cond_cols=[]
            #     if sep_set_for_indtest.any():
            #         for t_lag in range(0,lag):
            #             cond_col=sep_set_for_indtest[1+t_lag:-lag+t_lag, :]
            #             cond_cols.append(cond_col)    
            #         sep_set_all_for_indtest=np.hstack(cond_cols)
            #         # print(sep_set_all_for_indtest.shape)
            #         p_value = HSICtest.CIndtest(data[1+lag:, [node_to_col_idx[i]]].reshape(-1, 1), data[1+lag:, [node_to_col_idx[j]]].reshape(-1, 1), sep_set_all_for_indtest)[0]
            #     else:
            #         p_value = HSICtest.UIndtest(data[1+lag:, [node_to_col_idx[i]]].reshape(-1, 1), data[1+lag:, [node_to_col_idx[j]]].reshape(-1, 1))[0]

            if sep_set_for_indtest.any():
                    p_value = HSICtest.CIndtest(data[1+lag:, [node_to_col_idx[i]]].reshape(-1, 1), data[1+lag:, [node_to_col_idx[j]]].reshape(-1, 1), sep_set_for_indtest[1:-lag, :])[0]
            else:
                    p_value = HSICtest.UIndtest(data[2:, [node_to_col_idx[i]]].reshape(-1, 1), data[2:, [node_to_col_idx[j]]].reshape(-1, 1))[0]
                
            ## Linear simple causality
            threshold = 0.05

            ## Nonlinear complex causality
            # threshold = 1e-3

            if p_value<threshold:
                sep_set_i = [node for node in adjacency[i] if node != i]
                sep_set_j = [node for node in adjacency[j] if node != j]
                # condition sets for TE
                sep_set_ij= [data[1:-lag, [node_to_col_idx[k]]].reshape(-1, 1) for k in sep_set_i if k in node_to_col_idx]
                sep_set_ji= [data[1:-lag, [node_to_col_idx[k]]].reshape(-1, 1) for k in sep_set_j if k in node_to_col_idx]

                # Delta CTE
                TEij = kde.new_te(data[1:-lag, [node_to_col_idx[i]]], data[1+lag:, [node_to_col_idx[j]]], sep_set_ij, bandwidth)
                TEji = kde.new_te(data[1:-lag, [node_to_col_idx[j]]], data[1+lag:, [node_to_col_idx[i]]], sep_set_ji, bandwidth)
                delta_te = TEij - TEji

                # Significance test
                low_band, high_band = block_bootstrap(data[1:-lag, [node_to_col_idx[i]]], data[1+lag:, [node_to_col_idx[j]]], sep_set_ij, sep_set_ji, bandwidth, block_size=10, n_bootstrap=5000, low_per=2.5, high_per=97.5)
                if delta_te < low_band or delta_te > high_band:
                    if (delta_te > 0):
                        new_ad[i].append(j)
    return new_ad


# Significance test for differences in transfer entropy
def block_bootstrap(x,y,sep_set_x,sep_set_y,bandwidth,block_size,n_bootstrap,low_per,high_per):
    bootstrap_samples=[]
    for _ in range(n_bootstrap):
        blocks=[]
        for s in range(0,len(x),block_size):
            block_x = x[s:s+block_size]
            block_y = y[s:s+block_size]
            if len(block_x) < block_size:
                block_x = x[s:]
                block_y = y[s:]
            blocks.append((block_x,block_y))
        indices = np.random.choice(len(blocks), size=len(blocks), replace=False)  # Resample block-level indexes
        sample_blocks = [blocks[i] for i in indices]
        # Combine into new timing data
        sample_x = np.concatenate([block[0] for block in sample_blocks])
        sample_y = np.concatenate([block[1] for block in sample_blocks])
        # Calculate transfer entropy differences
        TExy = kde.new_te(sample_x, sample_y, sep_set_y , bandwidth)
        TEyx = kde.new_te(sample_y, sample_x, sep_set_x, bandwidth) 
        diff = TExy - TEyx
        bootstrap_samples.append(diff)
    bootstrap_samples = np.array(bootstrap_samples)
    q1 = np.percentile(bootstrap_samples, low_per)
    q3 = np.percentile(bootstrap_samples, high_per)
    return q1,q3
