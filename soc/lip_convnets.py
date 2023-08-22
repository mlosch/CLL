import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cayley_ortho_conv import Cayley, CayleyLinear
from block_ortho_conv import BCOP
from skew_ortho_conv import SOC

from custom_activations import *
from utils import conv_mapping, activation_mapping

class NormalizedLinear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight/weight_norm
        return F.linear(X, self.lln_weight if self.training else self.lln_weight.detach(), self.bias)

class Linear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        return super(Linear, self).forward(X)

class MLP(nn.Module):
    def __init__(self, features, activation_name):
        super(MLP, self).__init__()
        assert len(features) >= 2
        modules = []
        N = len(features)-2
        for i in range(len(features)-1):
            modules.append(nn.Linear(features[i], features[i+1]))
            if i < N:
                modules.append(activation_mapping(activation_name, features[i+1]))
                self.register_buffer('power_iterate_{}'.format(i), torch.rand(features[i]))
        self.last_weight = modules[-1].weight
        self.layers = nn.ModuleList(modules)

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.linear(x, weight=W)
            x_ = F.linear(xp, weight=W.transpose(1,0))
            x = x_ / torch.norm(x_, p=2)
        return x
    
    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.linear(x, weight=W)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def getK(self, num_iter):
        K = 1.0
        N = len(self.layers)-1
        for i, mod in enumerate(self.layers):
            if i >= N:
                # skip final layer
                break

            if not hasattr(mod, 'weight'):
                continue

            W = mod.weight
            # print(i, self.__getattr__('power_iterate_{}'.format(i)))
            if self.training:
                x = self.power_iteration(num_iter, W, self.__getattr__('power_iterate_{}'.format(i)))
                with torch.no_grad():
                    torch.add(x.detach(), 0.0, out=self.__getattr__('power_iterate_{}'.format(i)))
                sigma = self.spectral_value(W, x)
            else:
                sigma = self.spectral_value(W, self.__getattr__('power_iterate_{}'.format(i)))
            K = K * sigma

        return K

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        # print('X', X.shape)
        for mod in self.layers:
            # if hasattr(mod, 'weight'):
            #     print('W', mod.weight.shape)
            X = mod(X)
            # print(mod, 'X', X.shape)
        return X


class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_layer, activation_name, stride=1, kernel_size=3):
        super(LipBlock, self).__init__()
        self.conv = conv_layer(in_planes, planes*stride, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.activation = activation_mapping(activation_name, planes*stride)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
        
class LipConvNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, lln=False, soc_fc=True, L_multiplier=1.0, fc_mlp=[]):
        super(LipConvNet, self).__init__()        
        self.lln = lln
        self.in_planes = 3

        # self.L_multiplier = torch.nn.Parameter(
        self.L_multiplier = torch.full((block_size*5,), L_multiplier)
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3)
        self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3)
        self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3)
        self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=1)
        
        self.flatten = False
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        if num_classes == 200:
            # e.g. tiny-imagenet
            flat_features *= 4
        if len(fc_mlp) > 0:
            features = fc_mlp
            features.append(num_classes)
            features.insert(0, flat_features)
            self.last_layer = MLP(features, 'maxmin')
        elif self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)
        elif conv_name == 'cayley':
            self.last_layer = CayleyLinear(flat_features, num_classes)
        else:
            if not soc_fc:
                self.last_layer = Linear(flat_features, num_classes)
            else:
                kernel_size = 1
                # if num_classes == 200:
                #     flat_features = flat_features//4
                #     kernel_size = 2
                self.flatten = True
                self.last_layer = conv_layer(flat_features, num_classes, 
                                             kernel_size=kernel_size, stride=1)

    def _make_layer(self, planes, num_blocks, conv_layer, activation, 
                    stride, kernel_size):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, 
                                   stride, kernel_size))
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        i = 0
        for layer in self.layer1.children():
            x = layer(x) * self.L_multiplier[i]
            i+=1
        for layer in self.layer2.children():
            x = layer(x) * self.L_multiplier[i]
            i+=1
        for layer in self.layer3.children():
            x = layer(x) * self.L_multiplier[i]
            i+=1
        for layer in self.layer4.children():
            x = layer(x) * self.L_multiplier[i]
            i+=1
        for layer in self.layer5.children():
            x = layer(x) * self.L_multiplier[i]
            i+=1
        # x = self.layer1(x) * self.L_multiplier
        # x = self.layer2(x) * self.L_multiplier
        # x = self.layer3(x) * self.L_multiplier
        # x = self.layer4(x) * self.L_multiplier
        # x = self.layer5(x) * self.L_multiplier
        # print(x.shape)
        if self.flatten:
            x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], 1, 1)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x


class MoonsLipConvNet(LipConvNet):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=2, input_side=32, lln=False, soc_fc=True, L_multiplier=1.0, fc_mlp=[10, 20, 40, 40, 20, 10]):
        super(LipConvNet, self).__init__()
        self.lln = lln
        self.in_planes = 2

        self.L_multiplier = L_multiplier
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int


        layers = []
        fc_mlp.insert(0, 2)
        for i in range(len(fc_mlp)-1):
            print(fc_mlp[i], fc_mlp[i+1])
            layers.append(self._make_layer(fc_mlp[i], fc_mlp[i+1], conv_layer, activation))
        self.layers = nn.Sequential(*layers)

        if lln:
            self.last_layer = NormalizedLinear(fc_mlp[-1], 2)
        else:
            if soc_fc:
                self.last_layer = conv_layer(fc_mlp[-1], 2, kernel_size=1, stride=1)
            else:
                self.last_layer = Linear(fc_mlp[-1], 2)

    def _make_layer(self, in_planes, out_planes, conv_layer, activation):
        return LipBlock(in_planes, out_planes, conv_layer, activation, 1, 1)

    def forward(self, x):
        x = x[..., None, None]
        for layer in self.layers.children():
            x = layer(x) * self.L_multiplier
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x
