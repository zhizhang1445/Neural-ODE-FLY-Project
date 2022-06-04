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
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
import itertools
import pickle

from scipy.integrate import solve_ivp
from IPython import display 
device = torch.device('cpu')

def plot_func(func, diffeq = None, size = 100, domain = [-1, 1]):
    coord, H1a, H2a = get_data_func(func, size = size, domain = domain)


    if diffeq is None:
        plt.figure(figsize=[10,10])
        plt.streamplot(H1, H2, H1a, H2a, density=2, color = 'r')
        plt.streamplot(H1, H2, H1dot, H2dot, density=2, color = 'b')
        plt.show()
    else:
        x, y = coord
        dy2dt = diffeq(0,[x,y])
        plt.figure(figsize=[10,10])
        plt.streamplot(x, y, H1a, H2a, density=2, color = 'r')
        plt.streamplot(x, y, dy2dt[0,:, :], dy2dt[1,:, :], density=2, color = 'b')
        plt.show()

def plot_func_no_vectorized(func, diffeq = None, size = 100, domain = [-1, 1]):
    coord, H1a, H2a = get_data_func(func, size = size, domain = domain)
    x, y = coord

    h = np.zeros([2, size**2])
    coord = np.array(coord).reshape(2, size**2)
    
    for i in range(size**2):
        h[:,i] = diffeq(0, coord[:,i])

    coord = coord.reshape(2, size, size)
    h = h.reshape(2, size, size)

    plt.figure(figsize=[10,10])
    plt.streamplot(x, y, H1a, H2a, density=2, color = 'r')
    plt.streamplot(x, y, h[0,:,:], h[1,:,:], density=2, color = 'b')
    plt.show()


def plot_V(func, num = None, continuous = True):
    coord, h_mask = get_data_V(func, flip = continuous, num = num)
    x, y = coord
    if continuous:
        plt.imshow(h_mask,cmap=plt.get_cmap('hsv'), vmin = np.amin(h_mask),vmax = np.amax(h_mask))
    else:  
        plt.contourf(x, y, h_mask, cmap='Blues')
        plt.colorbar()
        plt.show()

def plot_Eigval(func, size = 20, n = -3, erode_size = 2):
    sc = 10**n
    coord, eigvec1, eigvec2 = get_data_eig(func, size, erode_size, eig_type = 'vector')
    x, y = coord
    u1, v1 = eigvec1
    u2, v2 = eigvec2
    plt.figure(figsize=[10,10])
    plt.quiver(x,y,u1,v1, color= 'b',angles='xy', scale_units='xy', scale=1/sc)
    plt.quiver(x,y,u2,v2, color='r',angles='xy', scale_units='xy', scale=1/sc)
    plt.show()
    
def get_data_func_Fly(func, H1, H2, size = 100, ma = None):
  dXdt = func(torch.Tensor(np.stack([H1, H2], -1).reshape(size*size, 2))).cpu().detach().numpy()
  dXdt = dXdt.reshape(size,size,2)
  dxdt_ma = np.ma.masked_array(dXdt[:,:,0], mask=ma)
  dydt_ma = np.ma.masked_array(dXdt[:,:,1], mask=ma)

  return dxdt_ma, dydt_ma

def get_data_V_Fly(func, H1, H2, size = 100, ma = None, num = None, flip = False, X = np.linspace(-0.2, 0.4, 100)):
  x, y = np.meshgrid(X, X)

  if num is not None:
    a = func.potential(num, torch.Tensor(np.stack([H1, H2], -1
    ).reshape(size * size, 2)))
  else:
    a = func.total_V(torch.Tensor(np.stack([H1, H2], -1).reshape(size*size, 2)))

  a = a.cpu().detach().numpy()
  h = a.reshape(size,size)
  if ma is None:
    return h

  h_mask = np.ma.masked_array(h, mask=ma)
  
  if flip:
    hmask_flip = np.flip(h_mask, 0)
    return hmask_flip
  else:
    return h_mask

def get_data_eig_Fly(func, H1, H2, erode_size = 2, eig_type = 'vector', ma = None, eig_scale = True, domain = [-1, 1]):

    def apply(M, func):
        tList1 = [func(m) for m in torch.unbind(M, dim=0)]
        eig_val = torch.stack(tList1, dim=0)
        return  eig_val

    def eig_vec(m):
        a, b = torch.symeig(m,True)
        return b

    def eig_val(m):
        a, b = torch.symeig(m,True)
        return a
    size = len(H1)
    h = func.metric(torch.Tensor(np.stack([H1, H2], -1).reshape(size * size, 2))).cpu().detach()
    h = h.reshape(size * size, 2, 2)

    if eig_type == 'value':
        values = apply(h, eig_val)
        h = values[:,0].reshape(size,size)

        if ma is None:
            return h
        else:
            h_ma = np.ma.masked_array(h, mask=ma)
            return h_ma

    if eig_scale:
        vec = apply(h, eig_vec)
        val = apply(h, eig_val)

        h1 = val[:,0]
        u1,v1 = vec[:,0,:].split(1,1)

        u1 = (u1[:,0]*h1).reshape(size, size)
        v1 = (v1[:,0]*h1).reshape(size, size)

        h2 = val[:,1]
        u2,v2 = vec[:,1,:].split(1,1)
        u2 = (u2[:,0]*h2).reshape(size, size)
        v2 = (v2[:,0]*h2).reshape(size, size)

    else:
        vec = apply(h, eig_vec)

        u1,v1 = vec[:,0,:].split(1,1)

        u1 = (u1).reshape(size, size)
        v1 = (v1).reshape(size, size)

        u2,v2 = vec[:,1,:].split(1,1)
        u2 = (u2).reshape(size, size)
        v2 = (v2).reshape(size, size)

    if ma is None:
        return (H1,H2), (u1, v1), (u2,v2)

    dim_mask_0, dim_mask_1 = np.shape(ma)
    zoom_size = size/dim_mask_0
    ma = zoom(ma, zoom_size, order = 1 ,mode = 'nearest')

    if erode_size != 1:
        kernel = np.ones((erode_size,erode_size),np.float32)
        invert_img = ~ma
        erode = cv2.erode(invert_img.astype(float),kernel,iterations = 1)
        ma = ~erode.astype(bool)

    u1m = np.ma.masked_array(u1, mask=ma)
    v1m = np.ma.masked_array(v1, mask=ma)
    u2m = np.ma.masked_array(u2, mask=ma)
    v2m = np.ma.masked_array(v2, mask=ma)
    return (H1, H2), (u1m, v1m), (u2m, v2m)