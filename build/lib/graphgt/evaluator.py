import numpy as np
from dictances import bhattacharyya
from scipy.stats import entropy, wasserstein_distance

def compute_kernel(x,y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    x_tile = x.reshape(x_size,1,dim)
    np.tile(x_tile,(1,y_size,1))
    y_tile = y.reshape(1,y_size,dim)
    np.tile(y_tile,(x_size,1,1))
    return np.exp(-np.mean((x_tile-y_tile)**2,axis = 2)/float(dim))

def compute_mmd(x,y):
    x_kernel = compute_kernel(x,x)
    # print(x_kernel)
    y_kernel = compute_kernel(y,y)
    # print(y_kernel)
    xy_kernel = compute_kernel(x,y)
    # print(xy_kernel)
    return np.mean(x_kernel)+np.mean(y_kernel)-2*np.mean(xy_kernel)

def compute_kld(x,y):
    return entropy(x,y)

def compute_emd(x,y):
    x = x.squeeze()
    y = y.squeeze()
    return wasserstein_distance(x,y)

# tester
#batch = 1000
#x = np.random.rand(batch,1)
#y_baseline = np.random.rand(batch,1)
#y_pred = np.zeros((batch,1))
#
#print('MMD baseline', compute_mmd(x,y_baseline))
#print('MMD prediction', compute_mmd(x,y_pred))
#print ('KLD', compute_kld(x,y_baseline))
#print ('EMB', compute_emd(x,y_baseline))



