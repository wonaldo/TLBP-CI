import numpy as np
import pickle
from algorithm import hsic_te
from metrics import adj_matrix, SHD_distance, count_accuracy, compute_f1




# ## Traffic
# data = np.load("./causaltime_gen_ver1.0/traffic/gen_data.npy")
# graph = np.load("./causaltime_gen_ver1.0/traffic/graph.npy")
# node_indices=np.arange(20)
# data_selected=data[:,:,node_indices]
# results=[]
# num_samples=480
# for i in range(num_samples):
#     sample=data_selected[i]
#     nodes=list(range(20))
#     nnodes=len(nodes)
#     adjacency,ad=hsic_te(sample,nodes,bandwidth=2.0,alpha=0.05)
#     results.append(ad)
# with open("CIT_BIT_Traffic.pkl",'wb') as file:
#     pickle.dump(results,file)

# shd_te_list=[]
# f1list_te=[]
# fdrlist_te=[]
# for m in range(len(results)):
#     adj_te=results[m]
#     shd_te = SHD_distance(graph, adj_matrix(20, adj_te)) # Calculate SHD
#     shd_te_list.append(shd_te)
#     f1_te = compute_f1(graph, adj_matrix(20, adj_te))  # Calculate F1
#     f1list_te.append(f1_te)
#     fdr_te = count_accuracy(graph, adj_matrix(20, adj_te)) # Calculate FDR
#     fdrlist_te.append(fdr_te)
# print("CIT-TBP: Mean = {:.4f}, Std = {:.4f}".format(np.mean(shd_te_list), np.std(shd_te_list)))


# ## Medical
# data=np.load("./causaltime_gen_ver1.0/medical/gen_data.npy")
# graph=np.load("./causaltime_gen_ver1.0/medical/graph.npy")
# node_indices=np.arange(20)
# data_selected=data[:,:,node_indices]
# results=[]
# num_samples=480
# for i in range(num_samples):
#     sample=data_selected[i]
#     nodes=list(range(20))
#     nnodes=len(nodes)
#     adjacency,ad=hsic_te(sample,nodes,bandwidth=2.0,alpha=0.05)
#     results.append(ad)
#     # print(i)
# with open("CIT_BIT_Medical.pkl",'wb') as file:
#     pickle.dump(results,file)

# shd_te_list=[]
# f1list_te=[]
# fdrlist_te=[]
# for m in range(len(results)):
#     adj_te=results[m]
#     shd_te = SHD_distance(graph, adj_matrix(20, adj_te)) # Calculate SHD
#     shd_te_list.append(shd_te)
#     f1_te = compute_f1(graph, adj_matrix(20, adj_te))  # Calculate F1
#     f1list_te.append(f1_te)
#     fdr_te = count_accuracy(graph, adj_matrix(20, adj_te)) # Calculate FDR
#     fdrlist_te.append(fdr_te)
# print("CIT-TBP: Mean = {:.4f}, Std = {:.4f}".format(np.mean(shd_te_list), np.std(shd_te_list)))





## PM2.5
data=np.load("./causaltime_gen_ver1.0/pm25/gen_data.npy")
graph=np.load("./causaltime_gen_ver1.0/pm25/graph.npy")
node_indices=np.arange(36)
data_selected=data[:,:,node_indices]
results=[]
num_samples=480
for i in range(num_samples):
    sample=data_selected[i]
    nodes=list(range(36))
    nnodes=len(nodes)
    adjacency,ad=hsic_te(sample,nodes,bandwidth=2.0,alpha=0.05)
    results.append(ad)
#    print(i)
with open("CIT_BIT_PM25.pkl",'wb') as file:
    pickle.dump(results,file)

shd_te_list=[]
f1list_te=[]
fdrlist_te=[]
for m in range(len(results)):
    adj_te=results[m]
    shd_te = SHD_distance(graph, adj_matrix(36, adj_te)) # Calculate SHD
    shd_te_list.append(shd_te)
    f1_te = compute_f1(graph, adj_matrix(36, adj_te))  # Calculate F1
    f1list_te.append(f1_te)
    fdr_te = count_accuracy(graph, adj_matrix(36, adj_te)) # Calculate FDR
    fdrlist_te.append(fdr_te)
print("CIT-TBP: Mean = {:.4f}, Std = {:.4f}".format(np.mean(shd_te_list), np.std(shd_te_list)))
