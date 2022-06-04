import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class GaussianFunc_ChevMetric(nn.Module):

  def __init__(self, dim = 2, nhidden_M = 8, nhidden_P = 8, n_V = 2, nlayer_M = 1):
    super(GaussianFunc_ChevMetric, self).__init__()

    self.n_V = n_V
    self.ndim = dim
    self.potential_list = nn.ModuleList([])

    self.th = nn.Tanh() 
    self.sg = nn.Sigmoid()
    self.softplus = nn.Softplus(100)
    self.elu = nn.ELU(inplace = True)
    self.relu = nn.ReLU(inplace = True)

    self.metric_list = nn.ModuleList([nn.Linear(dim, nhidden_M)])
    for i in range(nlayer_M):
      self.metric_list.append(nn.Linear(nhidden_M, nhidden_M))
    self.metric_list.append(nn.Linear(nhidden_M, 4))

    for i in range(n_V):
      self.potential_list.append(nn.Linear(dim, nhidden_P))
      self.potential_list.append(nn.Linear(nhidden_P, nhidden_P, bias = False))
      self.potential_list.append(nn.Linear(nhidden_P, 1, bias = False))

  def softSig(self, input):
    Amp = 0.3
    slope = 2
    output = 1 + torch.log(slope*self.softplus(input) + 1) + torch.tanh(-1*slope*self.softplus(-1*input))
    output = Amp*output
    return output

  def metric_nn(self, x):
    out = self.metric_list[0](x)
    out = self.elu(out)
    
    for i, nlayer in enumerate(self.metric_list[1:-1]):
      out = nlayer(out)
      out = self.elu(out)
    
    out = self.metric_list[-1](out)

    return out

  def metric(self, x):
    temp_dim = self.metric_nn(x).shape
    nn_out = self.metric_nn(x).reshape(temp_dim[:-1]+(2, 2))
    nn_transpose = nn_out.transpose(1, 2)
    chev_metric = torch.matmul(nn_out, nn_transpose)
    return chev_metric

  def potential(self, num, x, return_all= False):
    size = 3
    start = size*num
    end = size*(num+1)
    layers_V = self.potential_list[start:end]
    self.last_layers = layers_V

    out = layers_V[0](x)
    out = torch.pow(out, 2)
    out1 = out

    out = layers_V[1](out)
    out = torch.exp(out)
    out2 = out

    out = layers_V[-1](out)
    out3 = out

    if return_all:
      return out1, out2, out3 
    else:
      return out3

  def gradient(self, num, x):
    f1,f2,f3 = self.potential(num, x, True)
    layers_V = self.last_layers

    W3 = layers_V[-1].weight.data
    grad = W3

    W2 = layers_V[1].weight.data
    f2_prime = f2
    grad = torch.mm(grad*f2_prime, W2)

    W1 = layers_V[0].weight.data
    f1_prime = 2*layers_V[0](x)
    grad = torch.mm(grad*f1_prime, W1)
    return grad

  def total_grad(self, x):
    grad = self.gradient(0, x)
    for i in range(1,self.n_V):
      grad = grad + self.gradient(i, x)
    grad = grad/self.n_V
    return grad

  def total_V(self,x):
    out = self.potential(0,x)
    for i in range(1,self.n_V):
      out = out + self.potential(i,x)
    out = out/self.n_V
    return out

  def forward(self, x):
    grad = self.total_grad(x).unsqueeze(2)
    mag = self.metric(x)
    out = torch.matmul(mag, grad).squeeze()
    return out