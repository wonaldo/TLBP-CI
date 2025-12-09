# KDE for TE
import numpy as np
from sklearn.neighbors import KernelDensity

def new_te(i,j,sep_set,bandwidth):  
    Data =np.hstack((i[:-1], j[1:], j[:-1]))
    if sep_set:
        for sep in sep_set:
            Data = np.hstack((Data, sep[:-1]))  # Adding a set of conditional variables

    # Calculate the joint probability of all variables
    data_all = Data
    model = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)   
    model.fit(data_all)  
    log_dens_all = model.score_samples(data_all)  # Calculation of joint density
    pXtYnYbQ = np.exp(log_dens_all)  # Convert to actual probability density

    # Calculate the joint probability of Yb and Q
    data_YbQ = np.delete(Data,[0,1],axis=1)
    model = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)   
    model.fit(data_YbQ)  
    log_dens_YbQ = model.score_samples(data_YbQ)  # Calculation of joint density
    pYbQ = np.exp(log_dens_YbQ)  # Convert to actual probability density


    # Calculate the joint probability of Yn, Yb and Q
    data_YnYbQ = np.delete(Data,[0],axis=1)
    model = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)   
    model.fit(data_YnYbQ) 
    log_dens_YnYbQ = model.score_samples(data_YnYbQ)  # Calculation of joint density
    pYnYbQ = np.exp(log_dens_YnYbQ)  # Convert to actual probability density


    # Calculate the joint probability of Xt, Yb and Q
    data_XtYbQ = np.delete(Data,[1],axis=1)
    model = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)       
    model.fit(data_XtYbQ)
    log_dens_XtYbQ = model.score_samples(data_XtYbQ)  # Calculation of joint density
    pXtYbQ = np.exp(log_dens_XtYbQ)  # Convert to actual probability density

    # Calculation of entropy
    a = pXtYnYbQ*pYbQ
    b = pYnYbQ*pXtYbQ
    H = np.mean(np.log2(a/b))
    return H