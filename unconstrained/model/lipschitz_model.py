import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from lib import util
from lib.scheduler import Scheduler, ScheduledModule, PolynomialScheduler
from model import BaseModel
from model.metrics import Histogram
import model.layers as layers


def wrap_into_lipschitz_layer(value, norm_p, ignore_batchnorm=False, calibrate_outputs=False, demean=False):
    # wrap conv and linear layers into LipschitzLayerComputers
    if isinstance(value, nn.Sequential):
        value = SequentialLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, nn.Conv2d):
        value = Conv2dLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    elif isinstance(value, nn.Linear):
        value = LinearLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, layers.BasicResNetBlock) or isinstance(value, layers.BasicResNetBlockBNConv):
        value = ResNetBlockLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, nn.AvgPool2d):
        value = AvgPool2dLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, nn.AdaptiveAvgPool2d):
        value = AdaptiveAvgPool2dLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, layers.BNConv2d):
        value = BNConv2dLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, layers.BNLinear):
        value = BNLinearLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    if isinstance(value, layers.PreActBNConv2d):
        value = PreActBNConv2dLipschitzComputer(value, norm_p, calibrate_outputs=calibrate_outputs, demean=demean)
    elif isinstance(value, nn.modules.batchnorm._BatchNorm):
        # beta = 1.5
        # value = BatchNormLipschitzConstraint(value, norm_p, beta)
        value = BatchNormLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm, calibrate_outputs=calibrate_outputs, demean=demean)
    elif isinstance(value, nn.modules.normalization.GroupNorm):
        value = GroupNormLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm, calibrate_outputs=calibrate_outputs, demean=demean)
    return value


class LipschitzModel(BaseModel):
    def __init__(self, classifier_layer=None, p=2, 
        use_softmax=False, fix_last_layer_lip=False, 
        calibrate_outputs=False, demean=False, fc_WN=False, dim_calibrated_WN=False, ignore_batchnorm=False,
        converge_every_training_step=False, update_after_epoch=True):
        super(LipschitzModel, self).__init__()
        self.fix_last_layer_lip = fix_last_layer_lip
        self.norm_p = p
        self.classifier_layer = classifier_layer
        self.use_softmax = use_softmax
        self.calibrate_outputs = calibrate_outputs
        self.fc_WN = fc_WN
        self.dim_calibrated_WN = dim_calibrated_WN
        self.demean = demean
        assert demean == False or calibrate_outputs == False
        self.ignore_batchnorm = ignore_batchnorm
        # all layers are added in model_builder.build
        # self.lip_estimate = 0
        self.register_buffer('lip_estimate', torch.ones(1).cuda() * np.nan)
        self.converge_every_training_step = converge_every_training_step
        self.update_after_epoch = update_after_epoch

    def post_training_iteration_hook(self, **kwargs):
        if self.converge_every_training_step:
            with torch.no_grad():
                self.lipschitz_estimate(num_iter=-1)

    def post_training_hook(self, **kwargs):
        data_dict = dict(lipschitz_estimate=OrderedDict())
        with torch.no_grad():
            metrics = dict(lipschitz_estimate=OrderedDict())

            K0 = self.lipschitz_estimate(num_iter=0)

            # update lipschitz estimates
            # run until convergence
            Kt = K0
            diff = np.inf
            print('Number of lipschitz estimate iterations: 1')
            print('\tLipschitz estimate: {}'.format(Kt.item()))
            eps = 1.e-9
            num_iters = 2
            while diff > eps and num_iters < 10000:
                Kt1 = self.lipschitz_estimate(num_iter=1, update=self.update_after_epoch)
                diff = (Kt1-Kt).abs()
                Kt = Kt1
                num_iters += 1
                # if num_iters % 100 == 0:
                #     print(num_iters, Kt1.item(), diff.item())

            metrics['lipschitz_estimate']['global before convergence'] = K0
            metrics['lipschitz_estimate']['global'] = Kt #self.lipschitz_estimate(num_iter=1000)
            print('Number of lipschitz estimate iterations: {}'.format(num_iters))
            print('\tepsilon: {}'.format((Kt1-Kt).abs().item()))
            print('\tLipschitz estimate: {}'.format(Kt.item()))
            print('\tLipschitz estimate before convergence: {}'.format(K0.item()))

            Kloss = 1.0
            for mod in self.loss.values():
                if hasattr(mod, 'estimate'):
                    Kloss *= mod.estimate().max()
            metrics['lipschitz_estimate']['global output'] = Kt * Kloss

            for name, child in self.named_children():
                if isinstance(child, LipschitzLayerComputer):
                    lip_estimate = child.estimate(num_iter=0)
                    if type(lip_estimate) is tuple:
                        metrics['lipschitz_estimate'][name+'_main'] = lip_estimate[0]
                        metrics['lipschitz_estimate'][name+'_residual'] = lip_estimate[1]
                        data_dict['lipschitz_estimate'][name+'_main'] = lip_estimate[0]
                        data_dict['lipschitz_estimate'][name+'_residual'] = lip_estimate[1]
                    else:
                        metrics['lipschitz_estimate'][name] = lip_estimate
                        data_dict['lipschitz_estimate'][name] = lip_estimate

            metrics['weight_rank'] = OrderedDict()
            metrics['singular_value'] = OrderedDict()
            for name, module in self.named_modules():
                if hasattr(module, 'running_var'):
                    metrics['singular_value'][name+'.running_var'] = Histogram(module.running_var)
                if hasattr(module, 'weight') and module.weight is not None:
                    W = module.weight
                    W = W.view(W.shape[0], -1)
                    metrics['weight_rank'][name] = torch.matrix_rank(W)
                    _, s, _ = torch.svd(W, some=False, compute_uv=False)
                    metrics['singular_value'][name] = Histogram(s)

                # if name == self.classifier_layer:
                #     logit_norms = torch.norm(module.parent.weight, p=2, dim=1)
                #     for logit_idx, logit_norm in enumerate(logit_norms):
                #         assert logit_idx <= 99, 'string construction limited to two digits.'
                #         data_dict['logit_norm'] = logit_norm

        if len(data_dict) > 0 and 'save_path' in kwargs:
            # assert 'save_path' in kwargs
            assert 'epoch' in kwargs
            torch.save(data_dict, os.path.join(kwargs['save_path'], 'epoch_{}_post.pth'.format(kwargs['epoch'])))

        return metrics

    def _set_flag_recursively(self, module, attr, value):
        if isinstance(module, LipschitzLayerContainer):
            for name, mod in module.parent.named_modules():
                # print(name)
                self._set_flag_recursively(mod, attr, value)
        elif isinstance(module, LipschitzLayerComputer):
            # print(' --> set')
            setattr(module, attr, value)

    def __setattr__(self, name, value):
        # Adding a LipschitzLayer should be possible by default.
        if not isinstance(value, LipschitzLayerComputer):
            # Otherwise, if its a regular module, try to wrap it
            if isinstance(value, nn.Module):
                if name == self.classifier_layer:
                    if isinstance(value, nn.Linear) or isinstance(value, layers.USV_BNLinear):
                        value = ClassifierLipschitzComputer(value, self.norm_p, calibrate_outputs=self.fc_WN, use_softmax=self.use_softmax)
                    elif isinstance(value, nn.Conv2d):
                        value = SegmentationClassifierLipschitzComputer(value, self.norm_p, calibrate_outputs=self.fc_WN, use_softmax=self.use_softmax)
                    else:
                        raise RuntimeError('{} equals classifier_layer. Should be of type nn.Linear, but is: {}'.format(name, type(value)))
                else:
                    value = wrap_into_lipschitz_layer(value, self.norm_p, self.ignore_batchnorm, self.calibrate_outputs, self.demean)
                
                # self._set_flag_recursively(value, 'calibrate_outputs', self.calibrate_outputs)
        super(LipschitzModel, self).__setattr__(name, value)

    def lipschitz_estimate(self, num_iter, layer_name=None, update=False):
        # print('n={}, update={}, cur_est={}'.format(num_iter, update, self.lip_estimate.item()))
        if num_iter == 0 and not np.isnan(self.lip_estimate.item()):
            return self.lip_estimate
        if np.isnan(self.lip_estimate.item()):
            num_iter = 1

        K = 1.0
        # for name, module in self.named_modules():
        for name, child in self.named_children():
            if isinstance(child, LipschitzLayerComputer):
                Kchild = child.estimate(num_iter, update=update)
                K = child.compose(Kchild, K)

            if layer_name is not None and name == layer_name:
                return K

        if update:
            self.lip_estimate = K.detach()

        return K


class LipschitzConstrainedModel(LipschitzModel):
    def __init__(self, classifier_layer, p=2, betas=1.0, num_iter=1):
        super(LipschitzConstrainedModel, self).__init__(p=p)
        print(betas)
        assert type(betas) is int or type(betas) is float or isinstance(betas, dict) or isinstance(betas, Scheduler), 'Type of beta incorrect {}'.format(str(type(betas)))
        if type(betas) is int or type(betas) is float or isinstance(betas, Scheduler):
            self.betas = util.DefaultFallbackDict(fallback=betas)
        else:
            self.betas = betas
        self.num_iter = num_iter
        self.classifier_layer = classifier_layer

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            if name == self.classifier_layer:
                # raise NotImplementedError
                # value = LinearLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
                # value = ClassifierLipschitzComputer(value, self.norm_p, use_softmax=self.use_softmax)
                value = ClassifierLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.Conv2d):
                value = Conv2dLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.Linear):
                value = LinearLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.modules.batchnorm._BatchNorm):
                value = BatchNormLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, layers.BasicResNetBlock):
                value = ResNetBlockLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            else:
                value = wrap_into_lipschitz_layer(value, self.norm_p)

        super(LipschitzConstrainedModel, self).__setattr__(name, value)


class LipschitzLayerComputer(nn.Module):
    def __init__(self, parent_module, p, calibrate_outputs=False, demean=False, method='full'):
        super(LipschitzLayerComputer, self).__init__()
        self.parent = parent_module
        self.method = method
        self.norm_p = p
        self.calibrate_outputs = calibrate_outputs
        self.demean = demean
        self.dim_calibrated_WN = False
        if method == 'full':
            self.register_buffer('input_shape', torch.Tensor())

            if isinstance(self, LipschitzLayerContainer):
                pass
            elif isinstance(self.parent, nn.Conv2d):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1], 32, 32))
            elif isinstance(self.parent, nn.Linear):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1]))
            elif isinstance(self, ScalingLipschitzComputer):
                pass
            elif isinstance(self, AdaptiveAvgPool2dLipschitzComputer) or isinstance(self, AvgPool2dLipschitzComputer):
                pass
            elif isinstance(self, BNConv2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.conv.weight.shape[1], 32, 32))
            elif isinstance(self, BNLinearLipschitzComputer):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.linear.weight.shape[1]))
            elif isinstance(self.parent, layers.USV_BNLinear):
                pass
            elif isinstance(self, PreActBNConv2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1], 32, 32))
            elif isinstance(self, BatchNormLipschitzComputer):
                if self.parent.affine:
                    self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[0]))
                else:
                    self.register_buffer('power_iterate', torch.ones(1))
            elif isinstance(self, GradBatchNorm2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.ones(1))
            elif isinstance(self, ResNetBlockLipschitzComputer):
                pass
            else:
                raise RuntimeError('Don\'t know how to initialize dimensioniality of power iterate.')
        elif method == 'flattened':
            M, N = self.parent.weight.view(self.parent.weight.shape[0], -1).shape
            u = torch.randn(M)
            v = torch.randn(N)
            self.register_buffer('u', u/torch.norm(u,p=p))
            self.register_buffer('v', v/torch.norm(v,p=p))
            self.power_iterate = lambda: (self.u, self.v)
        else:
            raise RuntimeError('Undefined method {}'.format(method))
        self.register_buffer('lip_estimate', torch.ones(1).cuda() * np.nan)
        self.convergence_iterations = np.inf

        if hasattr(self.parent, 'weight') and self.parent.weight is not None:
            self.register_buffer('running_mean', torch.zeros(self.parent.weight.shape[0]))

    def check(self):
        pass

    def power_iteration(self, num_iter, W, running_power_iterate):
        if self.method == 'full':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def power_iteration_converge(self, W, running_power_iterate, eps=1.e-5, max_iter=10000):
        running_power_iterate = self.power_iteration(1, W, running_power_iterate)
        with torch.no_grad():
            sigma_t = self.spectral_value(W, running_power_iterate)
        diff = np.inf

        it = 1
        while diff > eps and it < max_iter:
            running_power_iterate = self.power_iteration(1, W, running_power_iterate)
            with torch.no_grad():
                sigma_t1 = self.spectral_value(W, running_power_iterate)
                diff = (sigma_t1 - sigma_t).abs()
                sigma_t = sigma_t1
            it += 1
        self.convergence_iterations = it
        return running_power_iterate

    def spectral_value(self, W, running_power_iterate):
        """
        Computes largest spectral value of weight matrix W, 
        using right-singular vector estimate: running_power_iterate
        """
        if self.method == 'full':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def estimate(self, num_iter, update=False):
        """
        Estimates upper Lipschitz constant of module
        """
        raise NotImplementedError

    def estimate_inverse(self, num_iter):
        """
        Estimates lower Lipschitz constant of module, by doing power iterate on inv(W'W)
        """
        raise NotImplementedError

    def compose(self, Kself, Kparent):
        """
        Produces composition of Lipschitz constant according to module graph.
        By default, take the product, which is valid for functional compositions as f(g(x)).
        """
        return Kself * Kparent

    def _setup(self, x):
        if self.method == 'full':
            if self.input_shape.numel() == 0:
                self.input_shape = torch.Tensor(list(x.shape)).int().to(x.device)
                self.input_shape[0] = 1
                print('{} :: New Input shape: {}'.format(str(self), self.input_shape.cpu().numpy().tolist()))
                self.power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

    def forward(self, x):
        self._setup(x)
        out = self.parent(x)
        if self.demean:
            if len(out.shape) == 4:
                if self.training:
                    batch_mean = torch.mean(out, dim=[0,2,3], keepdim=True)
                    with torch.no_grad():
                        torch.add(self.running_mean * 0.9, batch_mean.squeeze() * 0.1, out=self.running_mean)
                else:
                    batch_mean = self.running_mean[None, :, None, None]
                
                if not isinstance(self, ClassifierLipschitzComputer):
                    out = (out - batch_mean)
            else:
                if self.training:
                    batch_mean = torch.mean(out, dim=[0], keepdim=True)
                    with torch.no_grad():
                        torch.add(self.running_mean * 0.9, batch_mean.squeeze() * 0.1, out=self.running_mean)
                else:
                    batch_mean = self.running_mean[None, :]
                if not isinstance(self, ClassifierLipschitzComputer):
                    out = (out - batch_mean)

        if self.calibrate_outputs:
            K = torch.norm(self.parent.weight.view(self.parent.weight.shape[0], -1), dim=1, p=2)
            if self.dim_calibrated_WN:
                K = K * np.sqrt(self.parent.weight.shape[0])

            if len(out.shape) == 4:
                if self.training:
                    batch_mean = torch.mean(out, dim=[0,2,3], keepdim=True)
                    with torch.no_grad():
                        torch.add(self.running_mean * 0.9, batch_mean.squeeze() * 0.1, out=self.running_mean)
                else:
                    batch_mean = self.running_mean[None, :, None, None]
                
                if not isinstance(self, ClassifierLipschitzComputer):
                    out = (out - batch_mean) / K[None, :, None, None]
                else:
                    out = out / K[None, :, None, None]
            else:
                if self.training:
                    batch_mean = torch.mean(out, dim=[0], keepdim=True)
                    with torch.no_grad():
                        torch.add(self.running_mean * 0.9, batch_mean.squeeze() * 0.1, out=self.running_mean)
                else:
                    batch_mean = self.running_mean[None, :]
                if not isinstance(self, ClassifierLipschitzComputer):
                    out = (out - batch_mean) / K[None, :]
                else:
                    out = out / K[None, :]

        return out

    def __repr__(self):
        if self.calibrate_outputs:
            return 'LC-calibrated{'+str(self.parent)+'}'
        else:
            return 'LC{'+str(self.parent)+'}'


class LipschitzLayerContainer(LipschitzLayerComputer):
    pass



# ------------------------------------------------------------------
# Lipschitz estimation classes


class Conv2dLipschitzComputer(LipschitzLayerComputer):

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
        if self.norm_p == 2:
            sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        else:
            sigma = torch.norm(Wx, p=self.norm_p) / torch.norm(x, p=self.norm_p)
        return sigma

    def estimate(self, num_iter, update=False):
        if self.norm_p == 1:
            # operator norm equals absolute column sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=[0,2,3]))
        elif self.norm_p == 2:
            self.check()

            if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
                return self.lip_estimate
            elif num_iter == 0:
                num_iter = 1

            W = self.parent.weight
            if self.calibrate_outputs:
                w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2)
                if self.dim_calibrated_WN:
                    w_norm = w_norm * np.sqrt(self.parent.weight.shape[0])
                W = W / w_norm[:, None, None, None]

            if num_iter > 0:
                x = self.power_iteration(num_iter, W, self.power_iterate.clone(memory_format=torch.contiguous_format))
            else:
                x = self.power_iteration_converge(W, self.power_iterate.clone(memory_format=torch.contiguous_format))
            sigma = self.spectral_value(W, x)#.clone(memory_format=torch.contiguous_format))

            if update:
                with torch.no_grad():
                    torch.add(x.detach(), 0.0, out=self.power_iterate)
        elif np.isinf(self.norm_p):
            # operator norm equals absolute row sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=[1,2,3]))

        if update:
            self.lip_estimate = sigma.detach()

        return sigma


class LinearLipschitzComputer(LipschitzLayerComputer):

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.linear(x, weight=W)
            x_ = F.linear(xp, weight=W.transpose(1,0))
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x
    
    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.linear(x, weight=W)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, update=False):
        if self.norm_p == 1:
            # operator norm equals absolute column sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=0))
        elif self.norm_p == 2:
            self.check()

            if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
                return self.lip_estimate
            elif num_iter == 0:
                num_iter = 1

            W = self.parent.weight
            if self.calibrate_outputs:
                w_norm = torch.norm(W, dim=1, p=2, keepdim=True)
                if self.dim_calibrated_WN:
                    w_norm = w_norm * np.sqrt(W.shape[0])
                W = W / w_norm

            if num_iter > 0:
                x = self.power_iteration(num_iter, W, self.power_iterate.clone(memory_format=torch.contiguous_format))
            else:
                x = self.power_iteration_converge(W, self.power_iterate.clone(memory_format=torch.contiguous_format))
            sigma = self.spectral_value(W, x) #.clone(memory_format=torch.contiguous_format))

            if update:
                with torch.no_grad():
                    torch.add(x.detach(), 0.0, out=self.power_iterate)
        elif np.isinf(self.norm_p):
            # operator norm equals absolute row sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=1))

        if update:
            self.lip_estimate = sigma.detach()
        return sigma


class ClassifierLipschitzComputer(LinearLipschitzComputer):
    def __init__(self, *args, **kwargs):
        self.use_softmax = kwargs.pop('use_softmax', False)
        super(ClassifierLipschitzComputer, self).__init__(*args, **kwargs)

    def estimate(self, num_iter, update=False):
        # kW \in [num_classes x N]
        W = self.parent.weight
        if self.calibrate_outputs:
            w_norm = torch.norm(self.parent.weight.view(self.parent.weight.shape[0], -1), dim=1, p=2, keepdim=True)
            if self.dim_calibrated_WN:
                w_norm = w_norm * np.sqrt(W.shape[0])
            W = W/w_norm
        K_ij = torch.cdist(W, W, p=self.norm_p)        
        # K_ij \in [num_classes x num_classes]

        if self.use_softmax and torch.is_grad_enabled() and self.training:
            inds = torch.triu_indices(*K_ij.shape, offset=1)
            K_ij = K_ij[inds[0,:], inds[1,:]]  # select triu entries
            sigma = torch.sum(K_ij * F.softmax(K_ij, dim=0).detach())
            self.lip_estimate = torch.max(K_ij).detach()
        else:
            sigma = torch.max(K_ij)
            self.lip_estimate = sigma.detach()
        
        return sigma

    def __repr__(self):
        return 'Classifier_'+super(ClassifierLipschitzComputer, self).__repr__()


class ResNetBlockLipschitzComputer(LipschitzLayerContainer):
    def __init__(self, *args, ignore_batchnorm=False, **kwargs):
        calibrate_outputs = kwargs.pop('calibrate_outputs', False)
        super(ResNetBlockLipschitzComputer, self).__init__(*args, calibrate_outputs=False, **kwargs)
        for name, child in list(self.parent.named_children()):
            wrapped_child = wrap_into_lipschitz_layer(child, self.norm_p, calibrate_outputs=calibrate_outputs, ignore_batchnorm=ignore_batchnorm)
            self.parent.__setattr__(name, wrapped_child)

    def estimate(self, num_iter, update=False):
        Kmain = torch.ones_like(self.lip_estimate)
        Kresidual = torch.ones_like(self.lip_estimate)
        for name, child in self.parent.named_children():
            if not isinstance(child, LipschitzLayerComputer):
                continue

            if name == 'downsample':
                Kmain = child.estimate(num_iter, update=update)
            else:
                Kresidual = Kresidual * child.estimate(num_iter, update=update)

        return (Kmain, Kresidual)

    def compose(self, Kself, Kparent):
        assert type(Kself) is tuple and len(Kself) == 2
        Kmain, Kresidual = Kself

        if isinstance(self.parent, layers.PreActBasicResNetBlock):
            # preactivated blocks apply a normalization layer before downsampling
            if self.parent.has_norm_layers:
                Kbn1 = self.parent.bn1.estimate(num_iter=0)
                Kparent = self.parent.bn1.compose(Kbn1, Kparent)
        
        Kresidual = self.parent.conv1.compose(Kresidual, Kparent)
        if self.parent.downsample is not None:
            Kparent = self.parent.downsample.compose(Kmain, Kparent)

        # Lipschitz bounds add together for sum of two functions
        return Kresidual + Kparent


class SequentialLipschitzComputer(LipschitzLayerContainer):
    def __init__(self, *args, ignore_batchnorm=False, **kwargs):
        calibrate_outputs = kwargs.pop('calibrate_outputs', False)
        super(SequentialLipschitzComputer, self).__init__(*args, calibrate_outputs=False, **kwargs)
        for i in range(len(self.parent)):
            child = self.parent[i]
            wrapped_child = wrap_into_lipschitz_layer(child, self.norm_p, calibrate_outputs=calibrate_outputs, ignore_batchnorm=ignore_batchnorm)
            self.parent[i] = wrapped_child

    def estimate(self, num_iter, update=False):
        K = torch.ones_like(self.lip_estimate)
        for i in range(len(self.parent)):
            child = self.parent[i]
            if isinstance(child, LipschitzLayerComputer):
                Kchild = child.estimate(num_iter, update=update)
                K = child.compose(Kchild, K)
        return K


class ScalingLipschitzComputer(LipschitzLayerComputer):
    def _setup(self, x):
        if isinstance(self.parent, layers.MaxNormTo1):
            if self.parent.denominator is None:
                self.lip_estimate = torch.ones_like(self.lip_estimate)
            else:
                self.lip_estimate = 1./self.parent.denominator.detach()
        else:
            raise NotImplementedError

    def forward(self, x_in):
        x_out = self.parent(x_in)
        self._setup((x_in, x_out))
        return x_out

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate


class AvgPool2dLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('calibrate_outputs', False)
        super(AvgPool2dLipschitzComputer, self).__init__(*args, calibrate_outputs=False, **kwargs)
        self._is_setup = False        

    def _setup(self, x):
        if not self._is_setup:
            self._is_setup = True
            with torch.no_grad():
                Weight = torch.eye(x.shape[1])[:,:,None, None] * (
                    torch.ones(self.parent.kernel_size,self.parent.kernel_size)[None,None,:,:]) / (self.parent.kernel_size*self.parent.kernel_size)
                if np.isinf(self.norm_p):
                    if self.norm_p == 1:
                        # operator norm equals absolute column sum
                        factor = torch.max(Weight.abs().sum(dim=[0,2,3])).item()
                    else:
                        # operator norm equals absolute row sum
                        factor = torch.max(Weight.abs().sum(dim=[1,2,3])).item()
                else:
                    Weight = Weight.to(x.device)
                    x = torch.randn(1,x.shape[1], 32, 32).to(x.device)
                    x = self.power_iteration_converge(Weight, x)
                    # x = self.power_iteration(1000, Weight, x)
                    factor = self.spectral_value(Weight, x)
                self.lip_estimate = torch.Tensor([factor])
        self.lip_estimate = self.lip_estimate.to(x.device)

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.stride, padding=self.parent.padding)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate

    def forward(self, x):
        self._setup(x)
        return self.parent(x)


class AdaptiveAvgPool2dLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('calibrate_outputs', False)
        super(AdaptiveAvgPool2dLipschitzComputer, self).__init__(*args, calibrate_outputs=False, **kwargs)
        self.lip_estimate_dict = dict()
        assert self.parent.output_size == 1

    def forward(self, x_in):
        H, W = x_in.shape[-2:]

        if (H,W) not in self.lip_estimate_dict:
            with torch.no_grad():
                Weight = torch.eye(x_in.shape[1])[:,:,None, None] * (
                    torch.ones(H,W)[None,None,:,:]) / (H*W)
                if np.isinf(self.norm_p):
                    if self.norm_p == 1:
                        # operator norm equals absolute column sum
                        factor = torch.max(Weight.abs().sum(dim=[0,2,3])).item()
                    else:
                        # operator norm equals absolute row sum
                        factor = torch.max(Weight.abs().sum(dim=[1,2,3])).item()
                else:
                    Weight = Weight.to(x_in.device)
                    x = torch.randn(1,x_in.shape[1],H,W).to(x_in.device)
                    x = self.power_iteration_converge(Weight, x)
                    factor = self.spectral_value(Weight, x)

            self.lip_estimate_dict[(H,W)] = torch.Tensor([factor]).to(x_in.device)

        self.lip_estimate = self.lip_estimate_dict[(H,W)]

        return self.parent(x_in)

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=1, padding=0)
            x_ = F.conv_transpose2d(xp, weight=W, stride=1, padding=0)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=1, padding=0)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate


class PreActBNConv2dLipschitzComputer(Conv2dLipschitzComputer):
    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        bn = self.parent.bn
        conv = self.parent.conv

        for i in range(num_iter):
            xp = x
            xp = F.conv2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        bn = self.parent.bn
        conv = self.parent.conv

        Wx = F.conv2d(x, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))

        sigma = sigma * torch.max(bn.weight/torch.sqrt(bn.running_var+bn.eps))

        return sigma



class BNConv2dLipschitzComputer(Conv2dLipschitzComputer):
    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate

        conv = self.parent.conv
        bn = self.parent.bn

        bn_var = self.parent.cur_batch_var

        W_bn = torch.diag(bn.weight/torch.sqrt(bn_var+bn.eps))
        W_shape = W.shape
        W = torch.mm(W_bn.detach(), W.view(W_shape[0], -1)).view(W_shape)

        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate

        conv = self.parent.conv
        bn = self.parent.bn

        bn_var = self.parent.cur_batch_var

        W_bn = torch.diag(bn.weight/torch.sqrt(bn_var+bn.eps))
        W_shape = W.shape
        W = torch.mm(W_bn.detach(), W.view(W_shape[0], -1)).view(W_shape)

        Wx = F.conv2d(x, weight=W, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, update=False):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        cur_batch_var = self.parent.cur_batch_var
        if not self.training:
            self.parent.cur_batch_var =  self.parent.bn.running_var


        x = self.power_iteration(num_iter, self.parent.conv.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
        sigma = self.spectral_value(self.parent.conv.weight, x)#.clone(memory_format=torch.contiguous_format))

        # if self.training:
        if update:
            with torch.no_grad():
                torch.add(x.detach(), 0.0, out=self.power_iterate)

        if update:
            self.lip_estimate = sigma.detach()

        # invalidate current batch variance estimate
        self.parent.cur_batch_var = cur_batch_var

        return sigma


class BNLinearLipschitzComputer(LinearLipschitzComputer):

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate

        linear = self.parent.linear
        bn = self.parent.bn

        bn_var = self.parent.cur_batch_var

        W_bn = torch.diag(bn.weight/torch.sqrt(bn_var+bn.eps))
        W = torch.mm(W_bn.detach(), W)

        for i in range(num_iter):
            xp = F.linear(x, weight=W)
            x_ = F.linear(xp, weight=W.transpose(1,0))
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate

        linear = self.parent.linear
        bn = self.parent.bn

        bn_var = self.parent.cur_batch_var

        W_bn = torch.diag(bn.weight/torch.sqrt(bn_var+bn.eps))
        W = torch.mm(W_bn.detach(), W)

        Wx = F.linear(x, weight=W)

        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, update=False):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        cur_batch_var = self.parent.cur_batch_var
        if not self.training:
            self.parent.cur_batch_var =  self.parent.bn.running_var

        x = self.power_iteration(num_iter, self.parent.linear.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
        sigma = self.spectral_value(self.parent.linear.weight, x)#.clone(memory_format=torch.contiguous_format))

        if update:
            with torch.no_grad():
                torch.add(x.detach(), 0.0, out=self.power_iterate)

        if update:
            self.lip_estimate = sigma.detach()

        self.parent.cur_batch_var = cur_batch_var

        return sigma


class USV_BNLinearLipschitzComputer(LipschitzLayerComputer):
    def estimate(self, num_iter, **kwargs):
        if self.parent.bn is not None:
            cur_batch_var = self.parent.cur_batch_var
            if not self.training:
                self.parent.cur_batch_var =  self.parent.bn.running_var

            s_bn = self.parent.s.abs() / torch.sqrt(self.parent.cur_batch_var + self.parent.bn.eps)
            sigma = s_bn.max()

            self.parent.cur_batch_var = cur_batch_var
        else:
            sigma = self.parent.s.abs().max()
        
        self.lip_estimate = sigma.detach()

        return sigma


class BatchNormLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, ignore_batchnorm, **kwargs):
        super(BatchNormLipschitzComputer, self).__init__(*args, **kwargs)
        self.ignore_batchnorm = ignore_batchnorm

    def check(self):
        pass

    def estimate(self, num_iter, **kwargs):
        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        else:
            num_iter = 1


        if isinstance(self.parent, layers.ClippingBatchNorm2d) or isinstance(self.parent, layers.NoStdBatchNorm2d):
            eps = self.parent.eps
        else:    
            eps = self.parent.eps

        if self.parent.affine:            
            W = self.parent.weight / torch.sqrt(self.parent.running_var + eps)
        else:
            W = 1.0 / torch.sqrt(self.parent.running_var + eps)

        sigma = torch.max(W.abs())

        self.lip_estimate = sigma.detach()

        if self.ignore_batchnorm:
            return self.lip_estimate
        else:
            return sigma


class GroupNormLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, ignore_batchnorm, **kwargs):
        super(GroupNormLipschitzComputer, self).__init__(*args, **kwargs)
        self.ignore_batchnorm = ignore_batchnorm
        self.sigma = None

    def estimate(self, num_iter, **kwargs):
        if num_iter > 0:
            return self.sigma
        else:
            return self.lip_estimate

    def forward(self, x):
        assert x.shape[1]%self.parent.num_groups == 0
        G = self.parent.num_groups
        N, K, H, W = x.shape
        with torch.no_grad():
            group_var = x.view(N, G, -1).var(dim=2, unbiased=False)
            group_var = group_var[:, None].repeat(1, K//G).view(N, K)

        eps = self.parent.eps
        if self.parent.affine:
            W = self.parent.weight.unsqueeze(0) / torch.sqrt(group_var + eps)
        else:
            W = torch.sqrt(group_var + eps)

        self.sigma = torch.max(W.abs())
        self.lip_estimate = self.sigma.detach()

        x_out = self.parent(x)
        return x_out

            

class BlendBatchNormLipschitzComputer(BatchNormLipschitzComputer):
    def estimate(self, num_iter, **kwargs):
        sigma = super(BlendBatchNormLipschitzComputer, self).estimate(num_iter, **kwargs)
        sigma = self.parent.bn_blend * sigma + (1.0 - self.parent.bn_blend)
        self.lip_estimate = sigma.detach()
        return sigma


class GradBatchNorm2dLipschitzComputer(LipschitzLayerComputer):
    def _setup(self, x):
        pass

    def estimate(self, num_iter, **kwargs):
        return self.power_iterate.detach()


# ------------------------------------------------------------------
# Lipschitz constraining classes

class Conv2dLipschitzConstraint(Conv2dLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter):
        super(Conv2dLipschitzConstraint, self).__init__(parent_module, norm_p)

        self.beta = beta
        self.num_iter = num_iter

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

        self.register_buffer('normed_power_iterate', torch.Tensor())

        M, N = self.parent.weight.view(self.parent.weight.shape[0], -1).shape
        u = torch.randn(M)
        v = torch.randn(N)
        self.register_buffer('power_iterate_u', u/torch.norm(u,p=2))
        self.register_buffer('power_iterate_v', v/torch.norm(v,p=2))

    def _setup(self, x):
        need_setup = False
        if self.input_shape.numel() == 0:
            need_setup = True
        super(Conv2dLipschitzConstraint, self)._setup(x)
        if need_setup:
            self.normed_power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

    def estimate(self, num_iter, update=False):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        x = self.power_iteration(num_iter, self.parent.weight, self.normed_power_iterate)

        if update:
            torch.add(x.detach(), 0.0, out=self.normed_power_iterate)

        sigma = self.spectral_value(self.parent.weight, x.clone(memory_format=torch.contiguous_format))

        if isinstance(self.parent, layers.InputNormPreservingConv2d) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)

        if update:
            self.lip_estimate = sigma.detach()

        return sigma

    def forward(self, x):
        self._setup(x)

        with torch.no_grad():
            if self.training:
                iterate = self.power_iteration(self.num_iter, self.parent.weight_orig, self.power_iterate)
                torch.add(iterate, 0, out=self.power_iterate)
                iterate = iterate.clone(memory_format=torch.contiguous_format)
            else:
                iterate = self.power_iterate
        sigma = self.spectral_value(self.parent.weight_orig, iterate)

        # # SN-GAN Version, assuming flattened weight matrix
        # weight_mat = self.parent.weight_orig.view(self.parent.weight_orig.shape[0], -1)
        # u, v = self.power_iterate_u, self.power_iterate_v
        # if self.training:
        #     with torch.no_grad():
        #         for _ in range(self.num_iter):
        #             v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1.e-9, out=v)
        #             u = normalize(torch.mv(weight_mat, v), dim=0, eps=1.e-9, out=u)
        #         u = u.clone(memory_format=torch.contiguous_format)
        #         v = v.clone(memory_format=torch.contiguous_format)
        #         # self.power_iterate_u = u
        #         # self.power_iterate_v = v
        # sigma = torch.dot(u, torch.mv(weight_mat, v))

        if sigma.item() < self.beta:
            W = self.parent.weight_orig / 1.0
        else:
            W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)


class LinearLipschitzConstraint(LinearLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter):
        super(LinearLipschitzConstraint, self).__init__(parent_module, norm_p)

        self.beta = beta #nn.Parameter(torch.Tensor([beta]))
        self.num_iter = num_iter
        self.norm_p = norm_p

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

        self.register_buffer('normed_power_iterate', torch.Tensor())

    def _setup(self, x):
        need_setup = False
        if self.input_shape.numel() == 0:
            need_setup = True
        super(LinearLipschitzConstraint, self)._setup(x)
        if need_setup:
            self.normed_power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

    def estimate(self, num_iter, update=False):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        x = self.power_iteration(num_iter, self.parent.weight, self.normed_power_iterate)

        if update:
            torch.add(x.detach(), 0.0, out=self.normed_power_iterate)

        sigma = self.spectral_value(self.parent.weight, x.clone(memory_format=torch.contiguous_format))

        if isinstance(self.parent, layers.InputNormPreservingLinear) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)

        if update:
            self.lip_estimate = sigma.detach()

        return sigma

    def forward(self, x):
        self._setup(x)

        with torch.no_grad():
            if self.training:
                iterate = self.power_iteration(self.num_iter, self.parent.weight_orig, self.power_iterate)
                torch.add(iterate, 0, out=self.power_iterate)
                iterate = iterate.clone(memory_format=torch.contiguous_format)
            else:
                iterate = self.power_iterate
            sigma = self.spectral_value(self.parent.weight_orig, iterate)

        if sigma.item() < self.beta:
            W = self.parent.weight_orig / 1.0
        else:
            W = self.parent.weight_orig / 1.0
            W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)


class ClassifierLipschitzConstraint(LinearLipschitzConstraint):

    def estimate(self, num_iter, **kwargs):
        sigma = torch.max(torch.norm(self.parent.weight, p=self.norm_p, dim=1))
        self.lip_estimate = sigma.detach()
        return sigma

    def forward(self, x):
        with torch.no_grad():
            w_norm = torch.norm(self.parent.weight_orig, p=self.norm_p, dim=1)
            w_i, sigma = torch.max(w_norm, dim=0)

        if sigma.item() < self.beta:
            W = self.parent.weight_orig / 1.0
        else:
            W = self.parent.weight_orig / 1.0
            W[w_i, :] = self.beta * W[w_i, :] / sigma

        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'Classifier_'+super(ClassifierLipschitzConstraint, self).__repr__()


class BatchNormLipschitzConstraint(BatchNormLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter=1, **kwargs):
        super(BatchNormLipschitzConstraint, self).__init__(parent_module, norm_p, ignore_batchnorm=True, **kwargs)
        self.num_iter = num_iter
        self.beta = beta #nn.Parameter(torch.Tensor([beta]))
        self.norm_p = norm_p

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

    def forward(self, x):
        with torch.no_grad():
            if self.parent.affine:            
                W = self.parent.weight_orig / torch.sqrt(self.parent.running_var + self.parent.eps)
            else:
                W = 1.0 / torch.sqrt(self.parent.running_var + self.parent.eps)
            sigma = torch.max(W.abs())

        if sigma.item() < self.beta:
            W = self.parent.weight_orig / 1.0
        else:
            W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)


class BNLinearLipschitzConstraint(BNLinearLipschitzComputer, ScheduledModule):
    pass

class ResNetBlockLipschitzConstraint(ResNetBlockLipschitzComputer):
    def __init__(self, parent, norm_p, block_beta, num_iter, **kwargs):
        self.beta = block_beta
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                wrapped_child = Conv2dLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.Linear):
                wrapped_child = LinearLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, BNLinear):
                wrapped_child = BNLinearLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.modules.batchnorm._BatchNorm):
                wrapped_child = BatchNormLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.Sequential):
                wrapped_child = SequentialLipschitzConstraint(child, norm_p, block_beta, num_iter)
            else:
                wrapped_child = wrap_into_lipschitz_layer(child, norm_p)
            
            parent.__setattr__(name, wrapped_child)

        super(ResNetBlockLipschitzConstraint, self).__init__(parent, p=norm_p, **kwargs)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)

class SequentialLipschitzConstraint(SequentialLipschitzComputer):
    def __init__(self, parent, norm_p, block_beta, num_iter, **kwargs):
        self.beta = block_beta
        for i in range(len(parent)):
            child = parent[i]
            if isinstance(child, nn.Conv2d):
                wrapped_child = Conv2dLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.Linear):
                wrapped_child = LinearLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.modules.batchnorm._BatchNorm):
                wrapped_child = BatchNormLipschitzConstraint(child, norm_p, block_beta, num_iter)
            else:
                wrapped_child = wrap_into_lipschitz_layer(child, norm_p)
            parent[i] = wrapped_child

        super(SequentialLipschitzConstraint, self).__init__(parent, p=norm_p, **kwargs)

    def estimate(self, num_iter, update=False):
        K = torch.ones_like(self.lip_estimate)
        for i in range(len(self.parent)):
            child = self.parent[i]
            if isinstance(child, LipschitzLayerComputer):
                Kchild = child.estimate(num_iter, update=update)
                K = child.compose(Kchild, K)
        return K

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)
