import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.scheduler as scheduler


def layerclass_from_string(name):
    if name.startswith('nn.'):
        layerclass = nn.__dict__[name.replace('nn.','')]
    else:
        layerclass = sys.modules[__name__].__dict__[name]
    return layerclass


class BasicResNetBlock(nn.Module):
    """
    Adaptation of torchvision.resnet.BasicBlock version
    """
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        bias = None,
        nonlinearity = 'nn.ReLU',
        norm_layer = 'nn.BatchNorm2d'
    ):
        super(BasicResNetBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if norm_layer is not None:
            if isinstance(norm_layer, dict):
                norm_type = norm_layer.pop('type')
                norm_kwargs = norm_layer
                self.norm_layer_factory = lambda *args: layerclass_from_string(norm_type)(*args, **norm_kwargs)
            else:
                self.norm_layer_factory = layerclass_from_string(norm_layer)
            self.bn1 = self.norm_layer_factory(planes)
            self.bn2 = self.norm_layer_factory(planes)
            self.norm_is_affine = self.bn1.affine
            self.has_norm_layers = True
        else:
            self.has_norm_layers = False
            self.norm_is_affine = False

        if bias is not None:
            enable_bias = bias
        else:
            enable_bias = (self.norm_is_affine==False)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=enable_bias)

        if nonlinearity == 'nn.ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif nonlinearity == 'MaxMin':
            self.relu = MaxMin(planes//2)
        elif nonlinearity == 'nn.Sigmoid':
            self.relu = nn.Sigmoid()
        elif nonlinearity == 'nn.Tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=enable_bias)

        if inplanes != planes or stride != 1:
            downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=enable_bias)
            if self.has_norm_layers:
                downsample = nn.Sequential(
                    downsample,
                    self.norm_layer_factory(planes)
                    )
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride


    def forward(self, x):
        identity = x

        residual = self.conv1(x)
        if self.has_norm_layers:
            residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        if self.has_norm_layers:
            residual = self.bn2(residual)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(residual + identity)

        return out


class BasicResNetBlockBNConv(nn.Module):
    """
    Adaptation of torchvision.resnet.BasicBlock version
    """
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        nonlinearity = 'nn.ReLU',
    ):
        super(BasicResNetBlockBNConv, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = BNConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if nonlinearity == 'nn.ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif nonlinearity == 'MaxMin':
            self.relu = MaxMin(planes//2)
        elif nonlinearity == 'nn.Sigmoid':
            self.relu = nn.Sigmoid()
        elif nonlinearity == 'nn.Tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError
        self.conv2 = BNConv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        if inplanes != planes or stride != 1:
            downsample = BNConv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride


    def forward(self, x):
        identity = x

        residual = self.conv1(x)
        residual = self.relu(residual)

        residual = self.conv2(residual)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(residual + identity)

        return out


class PreActBasicResNetBlock(BasicResNetBlock):

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        nonlinearity = 'nn.ReLU',
        norm_layer = 'nn.BatchNorm2d'
    ):
        super(PreActBasicResNetBlock, self).__init__(inplanes, planes, stride, nonlinearity, norm_layer)
        if self.has_norm_layers:
            self.bn1 = self.norm_layer_factory(inplanes)
        if nonlinearity == 'MaxMin':
            self.relu1 = MaxMin(inplanes//2)
            self.relu2 = MaxMin(planes//2)
        else:
            self.relu1 = self.relu2 = self.relu
            self.relu = None
        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=(self.norm_is_affine==False))

    def forward(self, x):
        identity = x

        if self.has_norm_layers:
            x = self.bn1(x)
        x = self.relu1(x)
        residual = self.conv1(x)

        if self.has_norm_layers:
            residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = residual + identity

        return out

class PreActBNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity='nn.ReLU', *args, **kwargs):
        super(PreActBNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.bn = nn.BatchNorm2d(in_channels)
        if nonlinearity == 'nn.ReLU':
            kwargs = {'inplace': True}
            nonlin_type = nonlinearity
        elif isinstance(nonlinearity, dict):
            nonlin_type = nonlinearity.pop('type')
            kwargs = nonlinearity
        nonlin_class = layerclass_from_string(nonlin_type)
        self.relu = nonlin_class(**kwargs)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        # x = self.conv(self.bn(x))
        return x

    @property
    def weight(self):
        """
        For Lipschitz estimates only!
        """
        return self.conv.weight
        # if self.bn.affine:
        #     return self.conv.weight * (self.bn.weight[None, :, None, None] / torch.sqrt(self.bn.running_var[None, :, None, None] + self.bn.eps))
        # else:
        #     return self.conv.weight * (1.0 / torch.sqrt(self.bn.running_var[None, :, None, None] + self.bn.eps))

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation


class ScheduledFeatureAlphaDropout(nn.FeatureAlphaDropout, scheduler.ScheduledModule):
    def __init__(self, p, inplace=False) -> None:
        super(ScheduledFeatureAlphaDropout, self).__init__()
        self.p = p
        self.inplace = inplace


class USV_Module(nn.Module):
    @property
    def weight(self):
        raise NotImplementedError

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


class AddGloroAbsentLogit(scheduler.ScheduledModule):
    def __init__(self, output_module, epsilon, num_iter, lipschitz_computer, detach_lipschitz_computer=False, lipschitz_multiplier=1.0):
        super(AddGloroAbsentLogit, self).__init__()
        assert hasattr(output_module, 'parent'), 'output module must have attribute parent (as is defined in LipschitzLayerComputer.'
        self.W = lambda : output_module.parent.weight
        self.lc = lipschitz_computer
        self.detach_lipschitz_computer = detach_lipschitz_computer
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.lipschitz_multiplier = lipschitz_multiplier

    def forward(self, x):
        eps = self.epsilon

        K_lip = self.lipschitz_multiplier * self.lc(num_iter=self.num_iter)
        W = self.W()
        if self.detach_lipschitz_computer:
            K_lip = K_lip.detach()
            W = W.detach()
        return gloro_absent_logit(x, W, K_lip, epsilon=eps)

    def __repr__(self):
        if isinstance(self.epsilon, scheduler.Scheduler):
            return 'AddGloroAbsentLogit(eps={}, num_iter={})'.format(self.epsilon, self.num_iter)
        else:
            return 'AddGloroAbsentLogit(eps={:.2f}, num_iter={})'.format(self.epsilon, self.num_iter)


class PatchView(nn.Module):
    def forward(self, x):
        assert len(x.shape) == 5
        return x.view(x.shape[0]*x.shape[1], *x.shape[2:])


class BNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(BNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cur_batch_var = self.bn.running_var

    def forward(self, x, *args, **kwargs):
        x = self.conv(x)

        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=False)
            batch_var = ((x - batch_mean[None, :, None, None]) ** 2).mean(dim=[0, 2, 3], keepdim=False)
            self.cur_batch_var = batch_var

        x = self.bn(x)
        return x


class BNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BNLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.cur_batch_var = self.bn.running_var

    def forward(self, x, *args, **kwargs):
        x = self.linear(x)

        if self.training:
            batch_mean = x.mean(dim=0, keepdim=False)
            batch_var = ((x - batch_mean[None, :]) ** 2).mean(dim=0, keepdim=False)
            self.cur_batch_var = batch_var

        x = self.bn(x)
        return x


class Lip1GeLU(nn.Module):
    factor = 2./np.sqrt(np.pi)

    def forward(self, x):
        return F.gelu(x) / self.factor


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


class HouseHolder(nn.Module):
    def __init__(self, num_units):
        super(HouseHolder, self).__init__()
        # assert (channels % 2) == 0
        eff_channels = num_units
        # eff_channels = channels // 2
        
        self.theta = nn.Parameter(
                0.5 * np.pi * torch.ones(1, eff_channels, 1, 1).cuda(), requires_grad=True)

    def forward(self, z, axis=1):
        theta = self.theta
        x, y = z.split(z.shape[axis] // 2, axis)
                    
        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))
        
        a_2 = x*torch.cos(theta) + y*torch.sin(theta)
        b_2 = x*torch.sin(theta) - y*torch.cos(theta)
        
        a = (x * (selector <= 0) + a_2 * (selector > 0))
        b = (y * (selector <= 0) + b_2 * (selector > 0))
        
        return torch.cat([a, b], dim=axis)


class GroupSort(nn.Module):

    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.num_units, self.axis)
        # assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "

        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


class Flatten(nn.Module):
	def __init__(self, dim):
		super(Flatten, self).__init__()
		self.dim = dim

	def forward(self, x):
		return torch.flatten(x, self.dim)

	def __repr__(self):
		return 'Flatten(dim={})'.format(self.dim)


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1).max(-1)[0]
        # return torch.max(x, dim=self.dim)

    def __repr__(self):
        return 'Max(dim={})'.format(self.dim)


class PrintOutputShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

# -------------------------------------------------------------------------------------
# Helper functions

# def gloro_absent_logit(predictions, last_linear_weight, lipschitz_estimate, epsilon):
#     def get_Kij(pred, lc, W):
#         kW = W*lc
        
#         # with torch.no_grad():
#         y_j, j = torch.max(pred, dim=1)

#         # Get the weight column of the predicted class.
#         kW_j = kW[j]

#         # Get weights that predict the value y_j - y_i for all i != j.
#         #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
#         #kW_ij \in [256 x 128 x 10]
#         kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
        
#         K_ij = torch.norm(kW_ij, dim=1, p=2)
#         #K_ij \in [256 x 10]
#         return y_j, K_ij

#     #with torch.no_grad():
#     y_j, K_ij = get_Kij(predictions, lipschitz_estimate, last_linear_weight)
#     y_bot_i = predictions + epsilon * K_ij

#     # `y_bot_i` will be zero at the position of class j. However, we don't 
#     # want to consider this class, so we replace the zero with negative
#     # infinity so that when we find the maximum component for `y_bot_i` we 
#     # don't get zero as a result of all of the components we care aobut 
#     # being negative.
#     y_bot_i[predictions==y_j.unsqueeze(1)] = -np.infty
#     y_bot = torch.max(y_bot_i, dim=1, keepdim=True)[0]
#     all_logits = torch.cat([predictions, y_bot], dim=1)

#     return all_logits

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


def process_group_size(x, num_units, axis=-1):
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


def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))

    return sorted_x


def check_group_sorted(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)

    x_np = x.cpu().data.numpy()
    x_np = x_np.reshape(*size)
    axis = axis if axis == -1 else axis + 1
    x_np_diff = np.diff(x_np, axis=axis)

    # Return 1 iff all elements are increasing.
    if np.sum(x_np_diff < 0) > 0:
        return 0
    else:
        return 1