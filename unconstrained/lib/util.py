import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from itertools import chain, combinations


class DefaultFallbackDict(dict):
    def __init__(self, *args, **kwargs):
        self.fallback = kwargs.pop('fallback')
        super(DefaultFallbackDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            return self.fallback
        else:
            return super(DefaultFallbackDict, self).__getitem__(key)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class NullEnvironment(object):
    def __enter__(self):
        return
    def __exit__(self, *args):
        return


class Meter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0

    def update(self, val, n=None):
        self.val = val
        self.avg = val
        

class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def merge_dicts(x, y):
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = dict_of_dicts_merge(x[key], y[key])
    for key in x.keys() - overlapping_keys:
        # z[key] = deepcopy(x[key])
        z[key] = x[key]
    for key in y.keys() - overlapping_keys:
        z[key] = y[key]
        # z[key] = deepcopy(y[key])
    return z


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def domain_mask(domain, target):
    with torch.no_grad():
        mask = target == domain[0]
        for class_idx in domain[1:]:
            mask |= (target == class_idx)
    return mask


def fft_shift_matrix(n, s):
    shift = torch.arange(0, n).repeat((n, 1))
    shift = shift + shift.T
    return torch.exp(1j * 2 * np.pi * s * shift / n)

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

def cayley_conv(W):
    cout, cin, height, width = W.shape
    n = 32 # TODO
    s = (W.shape[2] - 1) // 2
    shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(W.device)
    # xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
    wfft = shift_matrix * torch.fft.rfft2(W, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
    W = cayley(wfft).view(cout, cin, height, width)
    return W


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """

    bn_init_parts = batchnorm.split('-')
    batchnorm = bn_init_parts[0]

    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif conv == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif conv == 'cayley':
                nn.init.kaiming_normal_(m.weight)
                with torch.no_grad():
                    m.weight = cayley_conv(m.weight) # orthogonalize
            elif conv == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif conv == 'normal1':
                nn.init.normal_(m.weight, 0.0, 1.0)
            elif conv == 'normalSqrtN':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)) and m.weight is not None:
            if batchnorm == 'normal':
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                nn.init.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            nn.init.constant_(m.bias, 0.0)

            if len(bn_init_parts) >= 2:
                with torch.no_grad():
                    if bn_init_parts[1] == 'L2normed':
                        m.weight /= torch.norm(m.weight, p=2)
                    elif bn_init_parts[1] == 'L1normed':
                        m.weight /= torch.norm(m.weight, p=1)
                    elif bn_init_parts[1] == 'Linfnormed':
                        m.weight /= torch.max(m.weight)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif linear == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif linear == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif linear == 'normal1':
                nn.init.normal_(m.weight, 0.0, 1.0)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        nn.init.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        nn.init.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    nn.init.constant_(param, 0)