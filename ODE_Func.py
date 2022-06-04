import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


class ODEFunc(nn.Module):

  def __init__(self, dim=2, nhidden_M=8, nhidden_P = 8, n_V = 2, 
               nlayer_M = 1, nlayer_V = 1):
    super(ODEFunc, self).__init__()
    self.ndim = dim
    self.nlayer_M = nlayer_M
    self.nlayer_V = nlayer_V
    self.n_V = n_V

    self.relu = nn.ReLU(inplace = True)
    self.th = nn.Tanh() 
    self.sg = nn.Sigmoid()
    self.elu = nn.ELU(inplace = True)

    self.metric_list = nn.ModuleList([nn.Linear(dim, nhidden_M)])
    for i in range(nlayer_M):
      self.metric_list.append(nn.Linear(nhidden_M, nhidden_M))
    self.metric_list.append(nn.Linear(nhidden_M, 4))

    self.potential_list = nn.ModuleList([])

    for i in range(n_V):
      self.potential_list.append(nn.Linear(dim, nhidden_P))
      for j in range(nlayer_V):
        self.potential_list.append(nn.Linear(nhidden_P, nhidden_P))
      self.potential_list.append(nn.Linear(nhidden_P, 1))

  def metric_nn(self, x):
    out = self.metric_list[0](x)
    out = self.elu(out)
    
    for i, nlayer in enumerate(self.metric_list[1:-1]):
      out = nlayer(out)
      out = self.elu(out)
    
    out = self.metric_list[-1](out)
    return out


  def metric(self, x):
    b1, b2, b3, b4 = self.metric_nn(x).split(1,1)
    m1 = b1**2 + b3**2
    m2 = b1*b2 + b3*b4
    m4 = b2**2 + b4**2
    
    m_matrix = torch.cat((m1,m2, m2,m4), dim = 1)
    return m_matrix

  def potential(self,num, t, x, return_all= False):
    size = self.nlayer_V + 2
    start = size*num
    end = size*(num+1)
    layers_V = self.potential_list[start:end]

    out2 = []
    self.last_layers = layers_V

    out = layers_V[0](x)
    out  = self.th(out)
    out1 = out

    for i, layer in enumerate(layers_V[1:-1]):
      out = layer(out)
      out = self.th(out)
      out2.append(out)

    out = layers_V[-1](out)
    #out = self.th(out)
    out = self.sg(out)
    out3 = out

    if return_all:
      return out1, out2, out3 
    else:
      return out3

  def total_V(self,t,x):
    out = self.potential(0,t,x)
    for i in range(1,self.n_V):
      out = out + self.potential(i,t,x)
    out = out/self.n_V
    return out

  def gradient(self,num, t, x):
    f1,f2,f3 = self.potential(num,t,x, True)
    layers_V = self.last_layers

    #f3_prime = 1-torch.pow(f3,2)
    f3_prime = self.sg(f3)*(1-self.sg(f3))
    W3 = layers_V[-1].weight.data
    grad = f3_prime*W3

    for i,layer in reversed(list(enumerate(layers_V[1:-1]))):
      W2 = layer.weight.data  
      f2_prime = 1-torch.pow(f2[i],2)
      grad = torch.mm(grad*f2_prime, W2)
    
    W1 = layers_V[0].weight.data
    f1_prime = 1-torch.pow(f1,2)
    grad = torch.mm(grad*f1_prime, W1)

    return grad

  def total_grad(self, t,x):
    grad = self.gradient(0,t,x)
    for i in range(1,self.n_V):
      grad = grad + self.gradient(i, t, x)
    grad = grad/self.n_V
    return grad

  def forward(self, t, x):
    grad = self.total_grad(t,x).repeat(1,2)
    mag = self.metric(x)
    out4d = mag*grad
    out2d1, out2d2 = out4d.split(2,1)
    out2d1 = out2d1.sum(1, keepdim=True)
    out2d2 = out2d2.sum(1, keepdim=True)
    out = torch.cat((out2d1, out2d2), dim = 1)
    return out