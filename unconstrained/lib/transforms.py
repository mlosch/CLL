import torch
import torch.nn as nn
import numpy as np

class PatchifyImage(nn.Module):
    def __init__(self, num_patches):
        super(PatchifyImage, self).__init__()
        self.num_patches = num_patches

    def forward(self, img):
        grid_n = int(np.sqrt(self.num_patches))
        assert img.shape[-1] % grid_n == 0
        patch_size = img.shape[-1] // grid_n
        img = torch.nn.functional.unfold(img.unsqueeze(0), patch_size, stride=patch_size).squeeze(0)
        img=img.view(3,img.shape[0]//3,img.shape[1])
        img=img.permute(2,0,1).contiguous()
        img=img.view(img.shape[0], img.shape[1], patch_size, patch_size)
        return img


class AddGaussianNoise(object):
    def __init__(self, p=0.5, mean=0., std=1., max_norm=1., value_range=(0, 1)):
        self.std = std
        self.mean = mean
        self.p = p
        self.max_norm = max_norm
        self.vrange = value_range

    def __call__(self, x):
        coinflip = (torch.rand(1) < self.p).item()
        if coinflip:
            delta = torch.empty_like(x).normal_()
            d_flat = delta.view(-1)
            n = d_flat.norm(p=2)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.max_norm
            img = torch.clamp(x + delta, self.vrange[0], self.vrange[1])
            return img
        else:
            return x
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0} mean={1}, std={2})'.format(self.p, self.mean, self.std)