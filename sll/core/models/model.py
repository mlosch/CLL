import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from core.models.layers import LinearNormalized, PoolingLinear, PaddingChannels, MaxMin
from core.models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class NormalizedModel(nn.Module):

  def __init__(self, model, mean, std):
    super(NormalizedModel, self).__init__()
    self.model = model
    self.register_buffer('mean', torch.Tensor(mean).reshape(1,3,1,1))
    self.register_buffer('std', torch.Tensor(std).reshape(1,3,1,1))
    # self.normalize = Normalize(mean, std)

  def forward(self, x):
    return self.model((x-self.mean)/self.std)
    # return self.model(self.normalize(x))


class LinearLipschitz(nn.Linear):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
      super(LinearLipschitz, self).__init__(in_channels, out_channels, *args, **kwargs)
      self.register_buffer('input_shape', torch.Tensor(1, in_channels))
      self.register_buffer('power_iterate', torch.randn(1, self.weight.shape[1]))
      self.activation = MaxMin(out_channels//2) #nn.ReLU(inplace=True)

    # def _setup(self, x):
    #     if self.input_shape.numel() == 0:
    #         print('setup', self.input_shape, x.shape)
    #         self.input_shape = torch.Tensor(list(x.shape)).int().to(x.device)
    #         self.input_shape[0] = 1
    #         # print('{} :: New Input shape: {}'.format(str(self), self.input_shape.cpu().numpy().tolist()))
    #         self.power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

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

    def estimate(self, num_iter, update=False):
        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        W = self.weight

        if num_iter > 0:
            x = self.power_iteration(num_iter, W, self.power_iterate.clone(memory_format=torch.contiguous_format))
        else:
            x = self.power_iteration_converge(W, self.power_iterate.clone(memory_format=torch.contiguous_format))
        sigma = self.spectral_value(W, x) #.clone(memory_format=torch.contiguous_format))

        if update:
            # self.power_iterate = x.detach()
            with torch.no_grad():
                torch.add(x.detach(), 0.0, out=self.power_iterate)

            self.lip_estimate = sigma.detach()
        return sigma

    def forward(self, x):
        # self._setup(x)
        return self.activation(super(LinearLipschitz, self).forward(x))

    def __repr__(self):
        return 'LC{'+str(super(LinearLipschitz, self).__repr__())+'}'


class LipschitzNetwork(nn.Module):

  def __init__(self, config, n_classes):
    super(LipschitzNetwork, self).__init__()

    self.depth = config.depth
    self.num_channels = config.num_channels
    self.depth_linear = config.depth_linear
    self.n_features = config.n_features
    self.conv_size = config.conv_size
    self.n_classes = n_classes

    if config.dataset == 'tiny-imagenet':
      imsize = 64
    else:
      imsize = 32

    self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

    layers = []
    block_conv = SDPBasedLipschitzConvLayer
    block_lin = SDPBasedLipschitzLinearLayer

    for _ in range(self.depth):
      layers.append(block_conv(config, (1, self.num_channels, imsize, imsize), self.num_channels, self.conv_size, layerK=config.unconstrained_layer_Ks))
    self.conv_layers = list(layers)

    layers.append(nn.AvgPool2d(4, divisor_override=4))
    self.stable_block = nn.Sequential(*layers)

    layers_linear = [nn.Flatten()]
    if config.dataset in ['cifar10', 'cifar100']:
      in_channels = self.num_channels * 8 * 8
    elif config.dataset == 'tiny-imagenet':
      in_channels = self.num_channels * 16 * 16

    self.soft_constrained_layers = []
    for di in range(self.depth_linear):
      if config.mlp and di == self.depth_linear-1:
        self.soft_constrained_layers.append(LinearLipschitz(in_channels, in_channels))
        layers_linear.append(self.soft_constrained_layers[-1])
      else:
        if config.liplin:
          self.soft_constrained_layers.append(LinearLipschitz(in_channels, in_channels))
          layers_linear.append(self.soft_constrained_layers[-1])
        else:
          layers_linear.append(block_lin(config, in_channels, self.n_features, layerK=config.unconstrained_layer_Ks))
    self.linear_layers = list(layers_linear[1:])

    if config.last_layer == 'pooling_linear':
      self.last_last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
    elif config.last_layer == 'lln':
      self.last_last = LinearNormalized(in_channels, self.n_classes)
    elif config.last_layer == 'vanilla':
      self.last_last = nn.Linear(in_channels, self.n_classes)
    else:
      raise ValueError("Last layer not recognized")

    self.layers_linear = nn.Sequential(*layers_linear)
    self.base = nn.Sequential(*[self.conv1, self.stable_block, self.layers_linear])
    
    self.unconstrained_layer_Ks = config.unconstrained_layer_Ks

  @property
  def Ks(self):
    if self.unconstrained_layer_Ks:
      ks = []
      for layer in self.conv_layers:
        ks.append(layer.k)
      for layer in self.linear_layers:
        ks.append(layer.k)
      ks = torch.cat(ks)
      print(torch.prod(ks), ks)
      return ks
    else:
      return None

  

  def forward(self, x):
    # if self.unconstrained_layer_Ks:
    #   x = self.conv1(x)
    #   layer_idx = 0
    #   for layer in self.stable_block:
    #     if isinstance(layer, SDPBasedLipschitzConvLayer):
    #       x = self.Ks[layer_idx] * layer(x)
    #       layer_idx += 1
    #     else:
    #       x = layer(x)

    #   for layer in self.layers_linear:
    #     if isinstance(layer, SDPBasedLipschitzLinearLayer):
    #       x = self.Ks[layer_idx] * layer(x)
    #       layer_idx += 1
    #     else:
    #       x = layer(x)

    #   return self.last_last(x)
    # else:
    return self.last_last(self.base(x))


