import os
import re
import itertools
from collections import OrderedDict
import struct

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

class Accuracy(nn.Module):
    def __init__(self, topk=1):
        super(Accuracy, self).__init__()
        self.topk = topk

    def forward(self, prediction, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = prediction.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)

    def __repr__(self):
        return 'Accuracy(topk={})'.format(self.topk)


class RobustAccuracy(nn.Module):
    def __init__(self, model, output_module, epsilon, data_std=[1,1,1], has_absent_logit=False, norm_p=2):
        super(RobustAccuracy, self).__init__()
        assert hasattr(output_module, 'parent'), 'output module must have attribute parent (as is defined in LipschitzLayerComputer.'
        self.W = lambda : output_module.parent.weight
        self.W_needs_calibration = output_module.calibrate_outputs
        self.output_module_lc = output_module.estimate
        self.lc = model.lipschitz_estimate
        self.epsilon = epsilon
        self.norm_p = norm_p
        self.data_std = torch.Tensor(data_std)
        self.has_absent_logit = has_absent_logit

    def forward(self, prediction, target):
        if self.has_absent_logit:
            prediction = prediction[:,:-1] # remove gloro logit

        # calculate lipschitz constants per logit
        eps = self.epsilon

        num_iter = 0 if self.training else -1

        with torch.no_grad():
            K_lip = 1./self.data_std.min()
            K_lip = K_lip * (self.lc(num_iter=num_iter, update=False) / self.output_module_lc(num_iter=num_iter, update=False))
            W = self.W()
            if self.W_needs_calibration:
                w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
                W = W/w_norm

            def get_Kij(pred, lc, W):
                kW = W*lc

                y_j, j = torch.max(pred, dim=1)

                # Get the weight column of the predicted class.
                kW_j = kW[j]

                # Get weights that predict the value y_j - y_i for all i != j.
                #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
                #kW_ij \in [256 x 128 x 10]
                kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
                
                K_ij = torch.norm(kW_ij, dim=1, p=self.norm_p)
                #K_ij \in [256 x 10]
                return y_j, j, K_ij

            y_j, pred_class, K_ij = get_Kij(prediction, K_lip, W)

            y_bot_i = prediction + eps * K_ij
            y_bot_i[prediction==y_j.unsqueeze(1)] = -np.infty
            y_bot = torch.max(y_bot_i, dim=1, keepdim=False)[0]

            robust = (pred_class == target) & (y_j > y_bot)

            batch_size = target.size(0)
            return robust.float().sum() * (100.0 / batch_size)


class RobustAccuracyV2(RobustAccuracy):

    def forward(self, prediction, target):
        # calculate lipschitz constants per logit
        eps = self.epsilon

        with torch.no_grad():
            K_lip = self.lc(num_iter=0) / self.output_module_lc(num_iter=0)
            W = self.W()
            K_logits = K_lip * torch.norm(W,dim=1,p=self.norm_p)  # num_classes

            y_j, j = torch.topk(prediction, k=2, dim=1)

            K_best = K_logits[j[:,0]]  # N (num_samples)
            K_second = K_logits[j[:,1]]  # N (num_samples)
            K_margin = (K_best+K_second) * eps

            y_bot = y_j[:,1] + K_margin # 2nd best logit + margin
            robust = (j[:,0] == target) & (y_j[:,0] > y_bot) # best logit still greater than y_bot?

            batch_size = target.size(0)
            return robust.float().sum() * (100.0 / batch_size)


class ConfusionMatrixMetric(nn.Module):
    def __init__(self, num_classes, robust_logit_index=None, labels=None):
        super(ConfusionMatrixMetric, self).__init__()
        self.num_classes = num_classes
        self.labels = labels
        self.robust_logit_index = robust_logit_index
        if robust_logit_index is not None:
            assert robust_logit_index == -1

    def forward(self, prediction, target):
        accuracies = []

        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.no_grad():
            if self.robust_logit_index is not None:
                pred = torch.max(prediction[:, :-1], dim=1)[1]
            else:
                pred = torch.max(prediction, dim=1)[1]
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        return ConfusionMatrix(confusion_matrix, self.labels)


class GloroCleanAccuracy(Accuracy):
    def forward(self, prediction, target):
        with torch.no_grad():
            # strip last logit
            return super(GloroCleanAccuracy, self).forward(prediction[:, :-1], target)

    def __repr__(self):
        return 'GloroCleanAccuracy(topk={})'.format(self.topk)


class LogitDistance(nn.Module):
    def forward(self, prediction):
        with torch.no_grad():
            N = prediction.shape[1]
            ntriu_elem = ((N*(N-1))/2)
            vals = prediction[:,:,None].repeat(1, 1, N)

            pairwise_diff = vals - vals.permute(0,2,1)
            triu = torch.triu(pairwise_diff)
            avg_diff = triu.abs().sum(dim=-1) / ntriu_elem

            return avg_diff.mean()


class MarginRatio(nn.Module):
    def __init__(self, data_std, reduce='mean'):
        super(MarginRatio, self).__init__()
        self.data_scaling = 1./np.min(data_std)
        self.reduce = reduce

    def forward(self, model, prediction, target):
        num_iter = 0 if self.training else -1

        with torch.no_grad():
            fc = model._modules[model.classifier_layer]
            Kfc = fc.estimate(num_iter=num_iter, update=False)
            W = fc.parent.weight
            if model._modules[model.classifier_layer].calibrate_outputs:
                w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
                W = W/w_norm
            K = (model.lipschitz_estimate(num_iter=num_iter, update=False) / Kfc) * self.data_scaling

            y_j, j = torch.topk(prediction, k=2, dim=1)

            kW = K * W
            kW_j = kW[j[:,0]]
            kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
            Kij = torch.norm(kW_ij, dim=1, p=2)

            margins = y_j[:,0].unsqueeze(1) - prediction  # N x Classes
            margins.scatter_(dim=1, index=j[:,0].unsqueeze(1), src=torch.full_like(margins, np.inf))
            ratios = margins / Kij
            ratio = ratios.min(dim=1)[0]

            if self.reduce == 'mean':
                return ratio.mean()
            elif self.reduce == 'min':
                return ratio.min()


class MarginRatioDistribution(nn.Module):
    def __init__(self, data_std):
        super(MarginRatioDistribution, self).__init__()
        self.data_scaling = 1./np.min(data_std)

    def forward(self, model, prediction, target):
        num_iter = 0 if self.training else -1

        with torch.no_grad():
            fc = model._modules[model.classifier_layer]
            Kfc = fc.estimate(num_iter=num_iter, update=False)
            W = fc.parent.weight
            if model._modules[model.classifier_layer].calibrate_outputs:
                w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
                W = W/w_norm
            K = (model.lipschitz_estimate(num_iter=num_iter, update=False) / Kfc) * self.data_scaling

            y_j, j = torch.topk(prediction, k=2, dim=1)

            kW = K * W
            kW_j = kW[j[:,0]]
            kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
            Kij = torch.norm(kW_ij, dim=1, p=2)

            margins = y_j[:,0].unsqueeze(1) - prediction  # N x Classes
            margins.scatter_(dim=1, index=j[:,0].unsqueeze(1), src=torch.full_like(margins, np.inf))
            ratios = margins / Kij #*(36./255.))
            ratios = ratios.min(dim=1)[0]
            return RunningHistogram(ratios)


class WeightRanks(nn.Module):
    def forward(self, model):
        if not self.training:
            return None

        ranks = OrderedDict()

        with torch.no_grad():
            for name, module in model.named_modules():
                if not hasattr(module, 'weight'):
                    continue

                W = module.weight
                W = W.view(W.shape[0], -1)

                ranks[name] = torch.matrix_rank(W)

class SingularValues(nn.Module):
    def forward(self, model):
        if not self.training:
            return None

        values = OrderedDict()

        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'running_var'):
                    values[name+'.running_var'] = Histogram(module.running_var)
                    
                if not hasattr(module, 'weight'):
                    continue

                W = module.weight

                if len(W.shape) == 1:
                    s = W
                else:
                    W = W.view(W.shape[0], -1)
                    _, s, _ = torch.svd(W, some=False, compute_uv=False)

                values[name] = Histogram(s)


class LayerOutputNorms(nn.Module):
    def __init__(self, dim, p=2, name_regex=None):
        super(LayerOutputNorms, self).__init__()
        self.p = p
        self.dim = tuple(dim)
        if name_regex is not None:
            self.regex = re.compile(name_regex)
        else:
            self.regex = None

    def forward(self, input, layer_output):
        norms = OrderedDict()
        norms['input'] = torch.norm(input, p=self.p, dim=self.dim).mean()

        for name, output in layer_output.items():
            if self.regex is not None:
                if self.regex.match(name) is None:
                    continue
            dim = self.dim
            if max(dim) >= output.ndim:
                dim = [d for d in dim if d < output.ndim]
            norms[name] = torch.norm(output, p=self.p, dim=dim).mean()

        return norms

    def __repr__(self):
        return 'LayerOutputNorms(p={}, dim={})'.format(self.p, self.dim)


class OutputDumper(nn.Module):
    def __init__(self, layer, save_path, every_epoch=1):
        super(OutputDumper, self).__init__()
        self.layer = layer
        self.save_path = save_path
        self.every_epoch = every_epoch

    def dump(self, tensor, fileprefix):
        filep = os.path.join(self.save_path, '{}.bin'.format(fileprefix))

        dims = max(1, tensor.numel() // tensor.shape[0])
        if not os.path.exists(filep):
            # first entry in file is dimensionality of output in big endian format (>)
            with open(filep, 'wb') as f:
                # value is saved as unsigned int
                f.write(struct.pack('>I', dims))

        with open(filep, 'ab') as f:
            for entry in tensor:
                entry = entry.numpy()
                fmt = {'float32': 'f', 'int32': 'i', 'uint32': 'I', 'int64': 'l', 'uint64': 'L'}[str(entry.dtype)]
                # convert float32 array to bytes in big endian format (>)
                if dims == 1:
                    s = struct.pack('>%s'%fmt, entry)
                else:
                    s = struct.pack('>%d%s'%(dims, fmt), *entry)
                f.write(s)

    def forward(self, layer_output, target):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.layer == 'target':
            self.dump(target.detach().cpu(), 'target')
        else:
            with torch.no_grad():
                output = layer_output[self.layer].detach()
            self.dump(output.cpu(), self.layer)
            
        

###############################################################################
## Container classes

class Histogram(object):
    def __init__(self, values):
        super(Histogram, self).__init__()
        self._values = values

    @property
    def values(self):
        return self._values

class RunningHistogram(Histogram):
    def __init__(self, values): #, normalization=1):
        super(RunningHistogram, self).__init__(values)

    def update(self, hist):
        self._values = torch.cat([self._values, hist._values], dim=0)

    @property
    def values(self):
        return self._values


class Plottable(object):
    def plot(self):
        raise NotImplementedError

class MatplotlibFigure(Plottable):
    def __init__(self, fig):
        self.fig = fig

    def plot(self):
        return self.fig

class ConfusionMatrix(Plottable):
    def __init__(self, values, labels=None):
        super(ConfusionMatrix, self).__init__()
        self._values = values
        self.labels = labels

    def update(self, mat):
        self._values += mat._values

    def plot(self):
        cm = self._values/self._values.sum(1)

        fig = plt.figure(0)
        _=plt.matshow(cm, fignum=0, cmap='Reds', vmin=0, vmax=1)
        _=plt.colorbar()

        N = cm.shape[0]

        if self.labels is not None:
            plt.xticks(torch.arange(N), self.labels)
            plt.yticks(torch.arange(N), self.labels)

        # Print values in cells
        for i, j in itertools.product(range(N), range(N)):
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")

        plt.xlabel('Predicted')
        plt.ylabel('True')

        return fig
