from pytorch_lightning import seed_everything
seed_everything(49)
import transfer
import importlib
import data_generator as dg
importlib.reload(transfer)

import HSICtest
importlib.reload(HSICtest)
import graph_generator as gg
importlib.reload(gg)
importlib.reload(dg)



def hsic_te(data,nodes,bandwidth,lag,alpha=0.05):
    adjacency = HSICtest.p4nodes(nodes,data,alpha)  
    new_ad = transfer.al4causal(nodes,data,bandwidth,adjacency,lag)
    return adjacency,new_ad