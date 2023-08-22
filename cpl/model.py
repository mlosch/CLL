import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from layers import *


class ConvexPotentialLayerNetwork(nn.Module):

    def __init__(self, depth, num_classes, depth_linear=3, conv_size=20, num_channels=20,
                 n_features=512, use_lln=True, use_vanilla_linear=False, block_lin='cpl'):

        super(ConvexPotentialLayerNetwork, self).__init__()
        self.num_classes = num_classes
        self.conv_size = conv_size
        self.depth = depth
        self.depth_linear = depth_linear
        self.num_channels = num_channels
        self.use_lln = use_lln
        self.use_vanilla_linear = use_vanilla_linear
        self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

        layers = []
        block_conv = ConvexPotentialLayerConv

        if block_lin == 'cpl':
            block_lin = ConvexPotentialLayerLinear
        elif block_lin == 'nn.Linear':
            block_lin = nn.Linear
        else:
            raise NotImplementedError(f'Unknown block type {block_lin}')

        for _ in range(self.depth):
            layers.append(block_conv((1, self.num_channels, 32, 32), self.num_channels, self.conv_size))
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        self.stable_block = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]
        for _ in range(self.depth_linear):
            layers_linear.append(block_lin(self.num_channels * 8 * 8, n_features))

        if self.use_lln:
            self.last_last = LinearNormalized(self.num_channels * 8 * 8, self.num_classes)
        elif self.use_vanilla_linear:
            self.last_last = nn.Linear(self.num_channels * 8 * 8, self.num_classes)
        else:
            self.last_last = PoolingLinear(self.num_channels * 8 * 8, self.num_classes, agg="trunc")

        self.layers_linear = nn.Sequential(*layers_linear)
        self.base = nn.Sequential(*[self.conv1, self.stable_block, self.layers_linear])

    def forward(self, x):
        # for layer in self.base.modules():
        #     x = layer(x)
        #     if hasattr(layer, 'weight'):
        return self.last_last(self.base(x))



