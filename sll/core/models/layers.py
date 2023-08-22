import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size

def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]

class MaxMin(nn.Module):
    def __init__(self, num_units, axis=1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=self.axis)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)

class SDPBasedLipschitzConvLayer(nn.Module):

  def __init__(self, config, input_size, cin, cout, kernel_size=3, epsilon=1e-6, layerK=False):
    super(SDPBasedLipschitzConvLayer, self).__init__()

    self.activation = nn.ReLU(inplace=False)

    self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.randn(cout))

    nn.init.xavier_normal_(self.kernel)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound) # bias init

    self.epsilon = epsilon
    self.k = None
    if layerK:
      self.k = nn.Parameter(torch.Tensor([1]))
    self.layerK = layerK

  def forward(self, x):
    kernel = self.kernel
    # if self.layerK:
    #   kernel = kernel*self.k
    res = F.conv2d(x, kernel, bias=self.bias, padding=1)
    res = self.activation(res)
    batch_size, cout, x_size, x_size = res.shape
    kkt = F.conv2d(kernel, kernel, padding=kernel.shape[-1] - 1)
    q_abs = torch.abs(self.q)
    T = 2 / (torch.abs(q_abs[None, :, None, None] * kkt).sum((1, 2, 3)) / q_abs)
    res = T[None, :, None, None] * res
    res = F.conv_transpose2d(res, kernel, padding=1)
    out = x - res
    return out  


class SDPBasedLipschitzLinearLayer(nn.Module):

  def __init__(self, config, cin, cout, epsilon=1e-6, layerK=False):
    super(SDPBasedLipschitzLinearLayer, self).__init__()

    self.activation = nn.ReLU(inplace=False)
    self.weights = nn.Parameter(torch.empty(cout, cin))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.rand(cout))

    nn.init.xavier_normal_(self.weights)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)  # bias init

    self.epsilon = epsilon
    if layerK:
      self.k = nn.Parameter(torch.Tensor([1]))
    self.layerK = layerK

  def forward(self, x):
    weights = self.weights
    # if self.layerK:
    #   weights = weights*self.k
    res = F.linear(x, weights, self.bias)
    res = self.activation(res)
    q_abs = torch.abs(self.q)
    q = q_abs[None, :]
    q_inv = (1/(q_abs+self.epsilon))[:, None]
    T = 2/torch.abs(q_inv * weights @ weights.T * q).sum(1)
    res = T * res
    res = F.linear(res, weights.t())
    out = x - res
    return out



class PaddingChannels(nn.Module):

  def __init__(self, ncout, ncin=3, mode="zero"):
    super(PaddingChannels, self).__init__()
    self.ncout = ncout
    self.ncin = ncin
    self.mode = mode

  def forward(self, x):
    if self.mode == "clone":
      return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
    elif self.mode == "zero":
      bs, _, size1, size2 = x.shape
      out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
      out[:, :self.ncin] = x
      return out


class PoolingLinear(nn.Module):

  def __init__(self, ncin, ncout, agg="mean"):
    super(PoolingLinear, self).__init__()
    self.ncout = ncout
    self.ncin = ncin
    self.agg = agg

  def forward(self, x):
    if self.agg == "trunc":
      return x[:, :self.ncout]
    k = 1. * self.ncin / self.ncout
    out = x[:, :self.ncout * int(k)]
    out = out.view(x.shape[0], self.ncout, -1)
    if self.agg == "mean":
      out = np.sqrt(k) * out.mean(axis=2)
    elif self.agg == "max":
      out, _ = out.max(axis=2)
    return out


class LinearNormalized(nn.Linear):

  def __init__(self, in_features, out_features, bias=True):
    super(LinearNormalized, self).__init__(in_features, out_features, bias)

  def forward(self, x):
    self.Q = F.normalize(self.weight, p=2, dim=1)
    return F.linear(x, self.Q, self.bias)


