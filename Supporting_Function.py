import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import cv2
import scipy
from skimage.feature import peak_local_max
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
import itertools
import pickle

from scipy.integrate import solve_ivp
from IPython import display 
device = torch.device('cpu')

def sample_Eq(n, eq, interval = [0, 1, 0, 1], t = 0):
    
    samp_coord = []
    samp_eq = []
    xinterval_len = interval[1]-interval[0]
    yinterval_len = interval[3]-interval[2]

    for i in range(0, n):
        x = xinterval_len*np.random.sample()+interval[0]
        y = yinterval_len*np.random.sample()+interval[2]
        coord = [x, y]
        samp_coord.append(coord)

        f = eq(t,coord)
        samp_eq.append(f)

    samp_coord = np.stack(samp_coord, axis=0)
    samp_eq = np.stack(samp_eq, axis=0)  
    return samp_coord, samp_eq


def generateData(n, diffeq, interval = [0,1], t_end = 0.25):
    t_eval = np.linspace(0, t_end , num=500)
    
    orig_y = []
    orig_t = t_eval
    samp_y = []
    samp_t = t_eval
    interval_len = interval[1]-interval[0]

    for i in range(0, n):
        x0 = interval_len*np.random.sample()+interval[0]
        y0 = interval_len*np.random.sample()+interval[0]
        z0 = [x0, y0]
        sol = solve_ivp(diffeq, [0, t_end ], z0, t_eval = t_eval)
        y = sol.y
        t = sol.t
        orig_y.append(y.T)
        samp_y.append(y.T)

    orig_y = np.stack(orig_y, axis=0)
    samp_y = np.stack(samp_y, axis=0)   
    return orig_y, samp_y, orig_t, samp_t

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def plot_params(params,length = 200, plot = False):
    A = []
    for i in range(len(params)):
        A = np.append(A,params[i].detach().numpy().reshape(-1))
    if plot:
        plt.figure()
        n = len(A)//length
    for i in range(n):
        B = A[i*length:i*length+length]
    if plot:
        plt.plot(np.arange(0,len(B),1), B)
    if plot:
        plt.plot(np.arange(0,len(A[length*n:]),1), A[length*n:])
        plt.show()
    return A

def extractData(H1dot, H1):
    size = 100
    V1 = H1dot.reshape(size*size, 1)
    N1 = H1.reshape(size*size, 1)
    N1 = N1[~V1.mask]
    V1  = V1[~V1.mask]
    N1 = N1[~np.isnan(V1)]
    V1 = V1[~np.isnan(V1)]
    N1 = np.expand_dims(N1, 1)
    V1 = np.expand_dims(V1, 1)
    return V1.data, N1

def index_map(h):
    f_1 = h[:,:,0]
    f_2 = h[:,:,1]
    norm = 1/(f_1**2 + f_2**2)

    G_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    G_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    grad_f_1 = scipy.signal.convolve2d(f_1, G_x, mode="same") + scipy.signal.convolve2d(f_1, G_y, mode="same")
    grad_f_2 = scipy.signal.convolve2d(f_2, G_x, mode="same") + scipy.signal.convolve2d(f_2, G_y, mode="same")
    index = norm*(f_1*grad_f_2 - f_2*grad_f_1)
    return index

def return_crit_point(indexmap):
    ind_peak = peak_local_max(indexmap, min_distance=2)
    ind_nan = np.argwhere(np.isnan(indexmap))
    ind_crit_point = np.concatenate((ind_peak, ind_nan), axis = 0)
    return ind_crit_point

def crit_point_stability(critpoint, h, fun_grad = None):
    if fun_grad is not None:
        f_1x, f_1y, f_2x, f_2y = fun_grad
    else:
        f_1 = h[:,:,0]
        f_2 = h[:,:,1]

        G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        f_1x, f_1y = scipy.signal.convolve2d(f_1, G_x, mode="same"), scipy.signal.convolve2d(f_1, G_y, mode="same")
        f_2x, f_2y = scipy.signal.convolve2d(f_2, G_x, mode="same"), scipy.signal.convolve2d(f_2, G_y, mode="same")

    x = critpoint[0]
    y = critpoint[1]
    A = np.array([[f_1x[x,y], f_1y[x,y]], [f_2x[x,y], f_2y[x,y]]])
    eigval = scipy.linalg.eig(A)[0]
    return np.real(eigval)

def index_to_coord(index_x, index_y, coord):
    x = coord[index_x, index_y, 0]
    y = coord[index_x, index_y, 1] 
    return x, y