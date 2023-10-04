import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.scheduler as scheduler
from model.lipschitz_model import LipschitzLayerComputer, LipschitzLayerContainer, ClassifierLipschitzComputer
import model.lipschitz_model as lip

import lib.util as util

def construct_loss(loss_cfg, default_reduction='sum'):
    if isinstance(loss_cfg, dict):
        loss_type = loss_cfg.pop('type')
    else:
        loss_type = loss_cfg
        loss_cfg = {}

    if loss_type.startswith('nn.'):
        loss_class = nn.__dict__[loss_type.replace('nn.','')]
    else:
        # print('__name__', __name__, type(__name__))
        current_module = __import__(__name__)
        # print('current_module', current_module, type(current_module))
        # print(current_module.__dict__.keys())
        loss_class = current_module.losses.__dict__[loss_type]
    if 'reduction' not in loss_cfg:
        loss_cfg['reduction'] = default_reduction
    return loss_class(**loss_cfg)

class GloroLoss(scheduler.ScheduledModule):
    def __init__(self, output_module, epsilon, num_iter, lipschitz_computer, alpha=0, K_scale=1.0, detach_Kfc=False, auto_deactivate=False, output_scale=1.0, reduction='mean', detach_lipschitz_computer=False, trades_lambda=0.0):
        super(GloroLoss, self).__init__()
        # raise DeprecatedError('Use layers.AddGloroAbsendLogit instead.')
        self.W = lambda : output_module.parent.weight
        self.W_needs_calibration = output_module.calibrate_outputs
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.reduction = reduction
        self.detach_lipschitz_computer = detach_lipschitz_computer
        self.alpha = alpha
        self.K_scale = K_scale
        self.auto_deactivate = auto_deactivate
        self.output_scale = output_scale
        self.detach_Kfc = detach_Kfc
        self.W_is_dim_calibrated = output_module.dim_calibrated_WN
        if self.W_is_dim_calibrated:
            raise NotImplementedError

        self.register_buffer('ij_constraint_count', torch.zeros(self.W().shape[0], self.W().shape[0]))
        self.trades_lambda = trades_lambda

    def forward(self, prediction, target):
        eps = self.epsilon
        K = self.lc(num_iter=self.num_iter, update=self.training)
        K = K / self.fc_lip_estimate()

        W = self.W()
        if self.W_needs_calibration:
            w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
            W = W/w_norm
        ij_pairs, loss = gloro_loss(prediction, target, W, K, epsilon=eps, reduction=self.reduction, K_blend=self.alpha, K_scale=self.K_scale, auto_deactivate=self.auto_deactivate, output_scale=self.output_scale, detach_K=self.detach_Kfc, trades_lambda=self.trades_lambda)

        # if self.training:
        #     with torch.no_grad():
        #         for i,j in ij_pairs:
        #             self.ij_constraint_count[i,j] += 1

        return loss

    def __repr__(self):
        if isinstance(self.epsilon, scheduler.Scheduler):
            return 'GloroLoss(eps={}, num_iter={})'.format(self.epsilon, self.num_iter)
        else:
            return 'GloroLoss(eps={:.2f}, num_iter={})'.format(self.epsilon, self.num_iter)

class LipschitzCrossEntropyLoss(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, num_iter, output_module, norm_p, 
        K_scale=1.0, alpha=1.0, detach_lipschitz_computer=True, detach_K=True, detach_Kfc=True, use_Kmax=False, 
        grad_scale=False, K_scale_is_parameter=False, add_epsilon=None, add_epsilon_one_sided=False, reduction='mean'):
        super(LipschitzCrossEntropyLoss, self).__init__()
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

        self.detach_K = detach_K
        self.detach_Kfc = detach_Kfc
        
        if detach_lipschitz_computer == False:
            assert detach_K == False or detach_Kfc == False

        if self.detach_K == False or self.detach_Kfc == False:
            assert detach_lipschitz_computer == False
            assert num_iter != 0

        if K_scale_is_parameter:
            self.K_scale = nn.Parameter(torch.Tensor([K_scale]))
        else:
            self.K_scale = K_scale
        self.use_Kmax = use_Kmax
        self.grad_scale = grad_scale
        self.alpha = alpha
        self.norm_p = norm_p
        assert isinstance(output_module, ClassifierLipschitzComputer) or isinstance(output_module, lip.ClassifierLipschitzConstraint)
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.W = lambda : output_module.parent.weight
        self.W_needs_calibration = output_module.calibrate_outputs
        self.W_is_dim_calibrated = output_module.dim_calibrated_WN

        self.add_epsilon = add_epsilon
        self.add_epsilon_one_sided = add_epsilon_one_sided

    @property
    def reduction(self):
        return self.loss.reduction

    @reduction.setter
    def reduction(self, value):
        self.loss.reduction = value

    def estimate(self, detach=True, target=None):
        W = self.W()
        if self.W_needs_calibration:
            w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
            if self.W_is_dim_calibrated:
                w_norm = w_norm * np.sqrt(W.shape[0])
            W = W/w_norm

        Kmin = 1.0
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter

        env = lambda : torch.no_grad() if detach else util.NullEnvironment()

        with env():
            if target is None:
                kW = torch.norm(W, p=self.norm_p, dim=1)
                if self.use_Kmax:
                    kW = torch.max(kW, keepdim=True)
                Ki = kW
            else:
                Wt = W[target]
                Wtj = Wt[:,:,None] - W.transpose(1,0).unsqueeze(0)
                kW_tj = torch.norm(Wtj, p=self.norm_p, dim=1)  # N x Classes

                kW = torch.norm(W, p=self.norm_p, dim=1) # Classes
                Ki = kW.unsqueeze(0) / kW_tj


        if not isinstance(self.K_scale, nn.Parameter):
            K_scale = max(self.K_scale, 1./Ki.min().item())
            Ki = Ki * K_scale
        else:
            # print(self.K_scale.detach().item())
            Ki = Ki * self.K_scale


        Ki = (self.alpha * Ki[None, :]) + (1.0 - self.alpha)
        # Ki = (self.alpha * (1.0/Ki[None, :])) + (1.0 - self.alpha)

        return Ki


    def add_offset(self, x, target, eps):
        with torch.no_grad():
            if self.add_epsilon_one_sided:
                delta = torch.zeros_like(x)
            else:
                delta = torch.full_like(x, eps)
            delta.scatter_(dim=1, index=target.unsqueeze(1), value=-eps)
        x = x + delta
        return x
    

    def forward(self, prediction, target):
        env = lambda : torch.no_grad() if self.detach_K else util.NullEnvironment()
        with env():
            K = self.lc(num_iter=self.num_iter, update=self.training) / self.fc_lip_estimate()
        # K = 1.0
        K_factor = self.estimate(detach=self.detach_Kfc, target=target)

        K_factor = K_factor * K
        # print(Ki.detach().min().item(), Ki.detach().max().item())
        
        weighted_prediction = (1./K_factor)*prediction

        if self.grad_scale:
            grad_scale = (self.alpha * Kmin) + (1.0 - self.alpha)
        else:
            grad_scale = 1.0

        if self.add_epsilon is not None:
            weighted_prediction = self.add_offset(weighted_prediction, target, self.add_epsilon)


        return grad_scale * self.loss(weighted_prediction, target)


class KeepLipschitzMarginCELoss(LipschitzCrossEntropyLoss):
    def __init__(self, min_class_distance, err_quantile=0.01, err_quantile_is_param=False, sigma=None, dist_type='logistic', freeze_logit_idx=None, *args, **kwargs):
        super(KeepLipschitzMarginCELoss, self).__init__(*args, **kwargs)
        # self.min_class_distance = min_class_distance #np.atleast_1d(min_class_distance)
        self.Di = min_class_distance
        # min_class_distance = np.array(min_class_distance)
        self.freeze_logit_idx = freeze_logit_idx

        self.dist_type = dist_type
        if dist_type == 'normal':
            from scipy.stats import norm
            self.icdf = norm.ppf
        elif dist_type == 'logistic':
            self.icdf = lambda y: np.log(y/(1-y))  #inverse logistic function = inverse cdf of logistic bell curve
        elif dist_type == 'hard_logistic':
            pass
        elif dist_type == 'raised_cosine':
            pass
        else:
            raise NotImplementedError

        self.sigma = None
        if sigma is not None:
            assert err_quantile is None
            self.sigma = sigma
            self.Q = None
            self.err_quantile_is_param = False
        else:
            if err_quantile == 0:
                raise AttributeError('err_quantile must be greater than 0')

            self.err_quantile = err_quantile

            if dist_type == 'hard_logistic':
                Q = 6.0
            elif dist_type == 'raised_cosine':
                Q = 1.0
            else:
                cdf_left = self.err_quantile/2.
                cdf_right = (1.0-self.err_quantile) + (self.err_quantile)/2.
                Q = self.icdf(cdf_right) - self.icdf(cdf_left)
            
            if err_quantile_is_param:
                self.Q = nn.Parameter(torch.ones(len(self.Di))*Q)
            else:
                self.Q = Q
            self.err_quantile_is_param = err_quantile_is_param

            print('sigma_i = ')
            print(self.Di / Q)

        # Di = min_class_distance

        # self.register_buffer('Di', torch.Tensor(Di))

    def estimate(self, detach=True, target=None):
        W = self.W()
        if self.W_needs_calibration:
            w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
            if self.W_is_dim_calibrated:
                w_norm = w_norm * np.sqrt(W.shape[0])
            W = W/w_norm

        Kmin = 1.0
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter

        env = lambda : torch.no_grad() if detach else util.NullEnvironment()

        with env():
            kW = W
            kW_i = kW
            Ki = torch.norm(kW, dim=1, p=self.norm_p)

        if self.sigma is None:
            if not self.err_quantile_is_param:
                if self.dist_type == 'hard_logistic':
                    Q = 6.0
                elif self.dist_type == 'raised_cosine':
                    Q = 1.0
                else:
                    cdf_left = self.err_quantile/2.
                    cdf_right = (1.0-self.err_quantile) + (self.err_quantile)/2.
                    Q = self.icdf(cdf_right) - self.icdf(cdf_left)
            else:
                Q = self.Q

            if self.dist_type == 'normal':
                scale = (self.Di / Q)**2
            else:
                scale = (self.Di / Q)
        else:
            scale = self.sigma

        K_scale = self.K_scale #max(self.K_scale, 1./Ktj.min().item())
        Ki = Ki * K_scale * scale

        return Ki

    def add_offset(self, x, target, eps):
        with torch.no_grad():
            delta = torch.full_like(x, eps)
            if self.add_epsilon_one_sided:
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=0)
            else:
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=-eps)

        if self.sigma is None:
            if not self.err_quantile_is_param:
                if self.dist_type == 'hard_logistic':
                    Q = 6.0
                elif self.dist_type == 'raised_cosine':
                    Q = 1.0
                else:
                    cdf_left = self.err_quantile/2.
                    cdf_right = (1.0-self.err_quantile) + (self.err_quantile)/2.
                    Q = self.icdf(cdf_right) - self.icdf(cdf_left)
            else:
                if np.random.rand() < 0.01:
                    print(self.Q.detach().item())
                Q = self.Q

            if self.dist_type == 'normal':
                sigma = (self.Di / Q)**2
            else:
                sigma = (self.Di / Q)
        else:
            sigma = self.sigma

        x = x + (delta/sigma)
        return x

    def forward(self, prediction, target):
        if self.freeze_logit_idx is not None:
            prediction[:, 0] = prediction[:, 0].detach()
        return super(KeepLipschitzMarginCELoss, self).forward(prediction, target)


class KeepLipschitzMarginCELossV2(KeepLipschitzMarginCELoss):
    pass


# class KeepLipschitzMarginBCELoss(KeepLipschitzMarginCELoss):
#     def __init__(self, *args, **kwargs):
#         super(KeepLipschitzMarginBCELoss, self).__init__(*args, **kwargs)
#         self.loss = BCEWithLogitsLoss()

# class KeepLipschitzMarginHardBCELoss(KeepLipschitzMarginCELoss):
#     def __init__(self, *args, **kwargs):
#         eps = kwargs.pop('clip_eps', 1.e-6)
#         super(KeepLipschitzMarginHardBCELoss, self).__init__(*args, **kwargs)
#         self.loss = BCEWithHardLogitsLoss(eps=eps)

# class KeepLipschitzMarginCosineBCELoss(KeepLipschitzMarginCELoss):
#     def __init__(self, *args, **kwargs):
#         eps = kwargs.pop('clip_eps', 1.e-6)
#         super(KeepLipschitzMarginCosineBCELoss, self).__init__(*args, **kwargs)
#         self.loss = BCEWithCosineLogitsLoss(eps=eps)

# class KeepLipschitzMarginHingeLoss(KeepLipschitzMarginCELoss):
#     def __init__(self, *args, **kwargs):
#         hinge_K = kwargs.pop('hinge_K', 1./kwargs['sigma'])
#         hinge_max_loss = kwargs.pop('hinge_max_loss', None)
#         super(KeepLipschitzMarginHingeLoss, self).__init__(*args, **kwargs)
#         self.loss = HingeLoss(eps=kwargs['add_epsilon'], K=hinge_K, max_loss=hinge_max_loss)


# class KeepPairwiseLipschitzMarginCELoss(LipschitzCrossEntropyLoss):
#     def __init__(self, pairwise_class_distance, err_quantile=0.01, dist_type='logistic', *args, **kwargs):
#         super(KeepPairwiseLipschitzMarginCELoss, self).__init__(*args, **kwargs)
#         pairwise_class_distance = np.array(pairwise_class_distance)

#         if dist_type == 'normal':
#             from scipy.stats import norm
#             icdf = norm.ppf
#         elif dist_type == 'logistic':
#             icdf = lambda y: np.log(y/(1-y))  #inverse logistic function = inverse cdf of logistic bell curve
#         else:
#             raise NotImplementedError
#         if err_quantile == 0:
#             raise AttributeError('err_quantile must be greater than 0')
#             #Q = icdf(1.0)
#         else:
#             cdf_left = err_quantile/2.
#             cdf_right = (1.0-err_quantile) + (err_quantile)/2.
#             Q = icdf(cdf_right) - icdf(cdf_left)

#         sigma_ij = pairwise_class_distance / Q
#         print('sigma_ij = ')
#         print(sigma_ij)
#         print('--------------------------')
#         # self.sigma_ij = torch.Tensor(sigma_ij).clamp(1.e-8)
#         self.register_buffer('sigma_ij', torch.Tensor(sigma_ij).clamp(1.e-8))
#         # print(self.sigma_ij)
#         pairwise_class_distance[np.eye(pairwise_class_distance.shape[0]).astype(np.bool)] = np.inf
#         closest_logit = pairwise_class_distance.argmin(axis=1)
#         # print(self.sigma_ij)
#         # print(closest_logit)
#         self.register_buffer('closest_logit', torch.Tensor(closest_logit).long())
#         # raise RuntimeError

#     def estimate(self, detach=True, target=None):
#         W = self.W()
#         if self.W_needs_calibration:
#             w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
#             if self.W_is_dim_calibrated:
#                 w_norm = w_norm * np.sqrt(W.shape[0])
#             W = W/w_norm

#         Kmin = 1.0
#         if not self.training:
#             num_iter = 0
#         else:
#             num_iter = self.num_iter

#         env = lambda : torch.no_grad() if detach else util.NullEnvironment()

#         with env():
#             kW = W
#             if target is not None:
#                 kW_t = kW[target]
#                 kW_tj = kW_t[:,:,None] - kW.transpose(1,0).unsqueeze(0)
#                 Ktj = torch.norm(kW_tj, dim=1, p=self.norm_p)
#                 # fill with 1s where t=j. Gradient will be 0 anyway when t=j
#                 Ktj = torch.scatter(Ktj, dim=1, index=target.unsqueeze(1), src=torch.ones_like(Ktj))
#                 sigma_tj = self.sigma_ij[target]

#                 # # take K only of closest logit
#                 # Ktj = torch.gather(Ktj, dim=1, index=self.closest_logit[target].unsqueeze(1))
#                 # sigma_tj = torch.gather(sigma_tj, dim=1, index=self.closest_logit[target].unsqueeze(1))

#             else:
#                 # determine all pairs
#                 Ktj = torch.cdist(kW, kW, p=self.norm_p)
#                 # Ktj[torch.eye(Ktj.shape[0], device=Kth.device)] = np.inf
#                 sigma_tj = self.sigma_ij

#         K_scale = self.K_scale #max(self.K_scale, 1./Ktj.min().item())
#         Ktj = Ktj * K_scale * sigma_tj

#         return Ktj

#     def forward(self, prediction, target):
#         if not self.training:
#             num_iter = 0
#         else:
#             num_iter = self.num_iter

#         env = lambda : torch.no_grad() if self.detach_K else util.NullEnvironment()
#         with env():
#             K = self.lc(num_iter=num_iter, update=self.training) / self.fc_lip_estimate()
#         Ktj = self.estimate(detach=self.detach_Kfc, target=target)

#         Ktj = Ktj * K
        
#         pred = prediction
        
#         target_logits = torch.gather(pred, dim=1, index=target.unsqueeze(1))
#         pairwise = torch.cat([target_logits.expand_as(pred).unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)
#         logsoftmax = torch.log_softmax(pairwise*(1.0/Ktj)[:,:,None], dim=-1)
        
#         # # # only minimize loss on logit pairs that are closest together in input space
#         # # to_min = torch.gather(logsoftmax, dim=1, index=self.closest_logit[target][:,None, None])
#         # # # print(to_min.shape)
#         # # return -to_min.mean()

#         # logsoftmax[:, :, 0].sum() would be enough to to the prediction, yet
#         #  we have double entries, which we'll remove
#         to_min = logsoftmax[:, :, 0].sum() - torch.gather(logsoftmax, dim=1, index=target[:, None, None]).sum()
#         return -to_min/(prediction.shape[0]*(prediction.shape[1]-1))


# class KeepPairwiseLipschitzMarginBCELoss(KeepPairwiseLipschitzMarginCELoss):
#     def __init__(self, *args, **kwargs):
#         super(KeepPairwiseLipschitzMarginBCELoss, self).__init__(*args, **kwargs)
    
#     def forward(self, prediction, target):
#         if not self.training:
#             num_iter = 0
#         else:
#             num_iter = self.num_iter

#         env = lambda : torch.no_grad() if self.detach_K else util.NullEnvironment()
#         with env():
#             K = self.lc(num_iter=num_iter, update=self.training) / self.fc_lip_estimate()
#         Ktj = self.estimate(detach=self.detach_Kfc, target=target)

#         Ktj = Ktj * K
        
#         pred = prediction
        
#         target_logits = torch.gather(pred, dim=1, index=target.unsqueeze(1))

#         target_logits = target_logits - prediction
#         logits = target_logits / Ktj

#         logprobs = F.logsigmoid(logits).clamp(-100.)  # clamp as described in PyTorch BCELoss
#         return -logprobs.mean()


class CLL(KeepLipschitzMarginCELoss):
    def __init__(self, *args, add_epsilon_one_sided=False, add_slack=None, **kwargs):
        super(CLL, self).__init__(*args, add_epsilon_one_sided=add_epsilon_one_sided, **kwargs)
        self.add_slack = add_slack

    def estimate(self, detach=True, target=None):
        W = self.W()
        if self.W_needs_calibration:
            w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
            if self.W_is_dim_calibrated:
                w_norm = w_norm * np.sqrt(W.shape[0])
            W = W/w_norm

        Kmin = 1.0
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter

        env = lambda : torch.no_grad() if detach else util.NullEnvironment()

        with env():
            kW = W

            if target is None:
                K_tj = torch.norm(kW, dim=1, p=2)
            else:
                kW_t = kW[target]
                kW_tj = kW_t[:,:,None] - kW.transpose(1,0).unsqueeze(0)
                K_tj = torch.norm(kW_tj, dim=1, p=2)
                K_tj = torch.scatter(K_tj, dim=1, index=target.unsqueeze(1), value=1.0)

        if self.sigma is None:
            if not self.err_quantile_is_param:
                if self.dist_type == 'hard_logistic':
                    Q = 6.0
                elif self.dist_type == 'raised_cosine':
                    Q = 1.0
                else:
                    cdf_left = self.err_quantile/2.
                    cdf_right = (1.0-self.err_quantile) + (self.err_quantile)/2.
                    Q = self.icdf(cdf_right) - self.icdf(cdf_left)
            else:
                Q = self.Q

            if self.dist_type == 'normal':
                scale = (self.Di / Q)**2
            else:
                scale = (self.Di / Q)
        else:
            scale = self.sigma

        K_scale = self.K_scale #max(self.K_scale, 1./Ktj.min().item())
        K_tj = K_tj * K_scale * scale

        return K_tj

    def add_offset(self, x, target, eps):
        with torch.no_grad():
            if self.add_epsilon_one_sided:
                delta = torch.full_like(x, eps)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=0)
            else:
                delta = torch.full_like(x, eps)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=-eps)

        if self.sigma is None:
            if not self.err_quantile_is_param:
                if self.dist_type == 'hard_logistic':
                    Q = 6.0
                elif self.dist_type == 'raised_cosine':
                    Q = 1.0
                else:
                    cdf_left = self.err_quantile/2.
                    cdf_right = (1.0-self.err_quantile) + (self.err_quantile)/2.
                    Q = self.icdf(cdf_right) - self.icdf(cdf_left)
            else:
                if np.random.rand() < 0.01:
                    print(self.Q.detach().item())
                Q = self.Q

            if self.dist_type == 'normal':
                sigma = (self.Di / Q)**2
            else:
                sigma = (self.Di / Q)
        else:
            sigma = self.sigma

        x = x + (delta/sigma)
        return x

    def forward(self, prediction, target):
        # if self.add_slack is not None:
        #     prediction = self.add_slack*torch.tanh(prediction/self.add_slack)

        env = lambda : torch.no_grad() if self.detach_K else util.NullEnvironment()
        with env():
            K = self.lc(num_iter=self.num_iter, update=self.training) / self.fc_lip_estimate()
        # K = 1.0
        K_factor = self.estimate(detach=self.detach_Kfc, target=target)

        K_factor = K_factor * K
        # print(Ki.detach().min().item(), Ki.detach().max().item())

        wx_i = prediction
        wx_t = torch.gather(wx_i, dim=1, index=target.unsqueeze(1))

        wx_it = wx_i - wx_t
        wx_it = (1./K_factor)*wx_it

        # if self.add_slack is not None:
        #     wrong_side = (wx_it) > 0  # N x Classes
        #     wx_it = wx_it - (wrong_side.float())*self.add_slack

        if self.grad_scale:
            grad_scale = (self.alpha * Kmin) + (1.0 - self.alpha)
        else:
            grad_scale = 1.0

        # if self.add_slack is not None:
        #     # wrong_side = (wx_it) > 0  # N x Classes
        #     # wx_it = wx_it - (wrong_side.float())*self.add_slack

        #     # wx_it[wx_it>0] = wx_it[wx_it] * self.add_slack
        #     wx_it = self.add_slack*torch.tanh(wx_it/self.add_slack)

        if self.add_epsilon is not None:
            wx_it = self.add_offset(wx_it, target, self.add_epsilon)



        return grad_scale * self.loss(wx_it, target)


class Identity(nn.Module):
    def forward(self, x, y):
        return x

# class TradesClipCE(CLL):
#     def __init__(self, *args, trades_lambda=1.0, **kwargs):
#         super(TradesClipCE, self).__init__(*args, **kwargs)
#         self.trades_lambda = trades_lambda
#         self.loss = Identity()

#     def trades_loss(self, logits, ref_logits, T=1.0):
#         prob_t = F.softmax(ref_logits/T, dim=1)
#         log_prob_s = F.log_softmax(logits, dim=1)
#         loss = -(prob_t*log_prob_s).sum(dim=1).mean()
#         return loss

#     def forward(self, prediction, target):
#         calib_logits = super(TradesClipCE, self).forward(prediction, target)

#         lam = self.trades_lambda
#         loss = F.cross_entropy(calib_logits, target) + lam * self.trades_loss(calib_logits, calib_logits.detach())
#         return loss


# class MarginLipschitzCrossEntropyLoss(scheduler.ScheduledModule):
#     def __init__(self, lipschitz_computer, num_iter, output_module, norm_p, K_scale=1.0, alpha=1.0, detach_lipschitz_computer=True, grad_scale=False, reduction='mean'):
#         super(MarginLipschitzCrossEntropyLoss, self).__init__()
#         self.lc = lipschitz_computer
#         self.num_iter = num_iter
#         self.loss = nn.CrossEntropyLoss(reduction=reduction)
#         self.detach_lipschitz_computer = detach_lipschitz_computer
#         self.K_scale = K_scale
#         self.grad_scale = grad_scale
#         self.alpha = alpha
#         self.norm_p = norm_p
#         assert isinstance(output_module, ClassifierLipschitzComputer)
#         self.fc_lip_estimate = lambda : output_module.lip_estimate
#         self.W = lambda : output_module.parent.weight
#         self.W_needs_calibration = output_module.calibrate_outputs
#         self.W_is_dim_calibrated = output_module.dim_calibrated_WN
#         if self.W_is_dim_calibrated:
#             raise NotImplementedError

#     @property
#     def reduction(self):
#         return self.loss.reduction

#     @reduction.setter
#     def reduction(self, value):
#         self.loss.reduction = value
    

#     def forward(self, prediction, target):
#         W = self.W()
#         if self.W_needs_calibration:
#             w_norm = torch.norm(W.view(W.shape[0], -1), dim=1, p=2, keepdim=True)
#             if self.W_is_dim_calibrated:
#                 w_norm = w_norm * np.sqrt(W.shape[0])
#             W = W/w_norm

#         if not self.training:
#             num_iter = 0
#         else:
#             num_iter = self.num_iter

#         if self.detach_lipschitz_computer:
#             with torch.no_grad():
#                 K = self.lc(num_iter=num_iter, update=self.training) / self.fc_lip_estimate()
#                 y_j, j = torch.topk(prediction, k=2, dim=1)

#                 kW = K * W
#                 kW_t = kW[target]
#                 kW_tj = kW_t[:,:,None] - kW.transpose(1,0).unsqueeze(0)
#                 Ktj = torch.norm(kW_tj, dim=1, p=2)
                
#                 y_t = torch.gather(prediction, dim=1, index=target.unsqueeze(1))
#                 margins = y_t - prediction  # N x Classes
#                 # first fill with inf where t==j
#                 margins.scatter_(dim=1, index=target.unsqueeze(1), src=torch.full_like(margins, np.inf))
#                 ratios = margins / Ktj
                
#                 # get index of smallest ratio per sample
#                 min_ratios, min_ratios_idx = ratios.min(dim=1)
#                 Kt_min = torch.gather(Ktj, dim=1, index=min_ratios_idx.unsqueeze(1))

#                 # then fill with min(margins/Ktj)
#                 # print(Ktj.shape, target.shape, Kt_min.shape)
#                 Ktj = torch.scatter(Ktj, dim=1, index=target.unsqueeze(1), src=Kt_min.expand_as(Ktj))
#                 # print(Ktj.shape, prediction.shape)
#         else:
#             raise RuntimeError

#         Ktj = torch.clamp(Ktj * self.K_scale, 1.0)
#         # print(Ki.detach().min().item(), Ki.detach().max().item())
#         blend_factor = (self.alpha * (1.0/Ktj)) + (1.0 - self.alpha)
#         weighted_prediction = blend_factor*prediction

#         if self.grad_scale:
#             grad_scale = (self.alpha * Kmin) + (1.0 - self.alpha)
#         else:
#             grad_scale = 1.0
#         return grad_scale * self.loss(weighted_prediction, target)



class ScaledCrossEntropyLoss(scheduler.ScheduledModule):
    def __init__(self, target, grad_scale=False, reduction='mean'):
        super(ScaledCrossEntropyLoss, self).__init__()
        self.target = target
        self.grad_scale = grad_scale
        self.reduction = reduction

    def estimate(self, detach=False):
        # K = self.lc(num_iter=num_iter, update=self.training)
        return torch.Tensor([1./self.target]).cuda()

    def forward(self, model, prediction, target):
        # with torch.no_grad():
        #     K = model.lipschitz_estimate(num_iter=0).item()
        # blend_factor = (self.scaling * (1.0/min(K, self.target))) + (1.0 - self.scaling)
        # if self.grad_scale:
        #     grad_scale = blend_factor
        # else:
        #     grad_scale = 1.0
        return F.cross_entropy(prediction/self.target, target, reduction=self.reduction)



# class BCEWithLogitsLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(BCEWithLogitsLoss, self).__init__()
#         self.loss = nn.BCEWithLogitsLoss(*args, **kwargs)

#     def forward(self, prediction, target):
#         onehot = torch.zeros_like(prediction)
#         onehot.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())
        
#         return self.loss(prediction, onehot)

# class BCEWithHardLogitsLoss(nn.Module):
#     def __init__(self, *args, eps=1.e-6, **kwargs):
#         super(BCEWithHardLogitsLoss, self).__init__()
#         self.loss = nn.BCELoss(*args, **kwargs)
#         self.eps = eps

#     def forward(self, prediction, target):
#         onehot = torch.zeros_like(prediction)
#         onehot.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())
        
#         prediction = F.hardsigmoid(prediction)
#         prediction = prediction.clamp(self.eps, 1.0-self.eps)
#         return self.loss(prediction, onehot)

# class BCEWithCosineLogitsLoss(nn.Module):
#     def __init__(self, *args, eps=1.e-6, **kwargs):
#         super(BCEWithCosineLogitsLoss, self).__init__()
#         self.loss = nn.BCELoss(*args, **kwargs)
#         self.eps = eps
#         # print('BCEWithCosineLogitsLoss :: clip_ep = {}'.format(eps))

#     def forward(self, prediction, target):
#         onehot = torch.zeros_like(prediction)
#         onehot.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())

#         x = prediction
#         probs = 0.5 * (1.0 + x + 1./np.pi * torch.sin(x*np.pi))
#         probs = probs.clamp(self.eps, 1.0-self.eps)
#         # print(probs)
#         return self.loss(probs, onehot)


# class HingeLoss(scheduler.ScheduledModule):
#     def __init__(self, K=1, eps=1, max_loss=None):
#         super(HingeLoss, self).__init__()
#         self.eps = eps
#         self.K = K
#         self.max_loss = max_loss

#     def forward(self, prediction, target):
#         nlogits = prediction.shape[1]
#         t = torch.full_like(prediction, -1.0) #/float(nlogits))
#         t.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())

#         loss = self.K*(self.eps - prediction*t)
#         if self.max_loss is None:
#             loss = torch.clamp(loss, 0)
#         else:
#             loss[prediction*t < -self.eps] = self.K*self.max_loss
#             # loss = torch.clamp(loss, 0, self.max_loss)
#         # print(self.K, loss.max().detach().item())
#         return loss.mean()

#     def __repr__(self):
#         return 'HingeLoss(eps={:.3f}, K={:.3f})'.format(self.eps, self.K)


class GlobalLipschitzDecay(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, lambda_, num_iter, pow=1.0, ignore_fc=False, classifier_each_logit=False, integrate_loss=None):
        super(GlobalLipschitzDecay, self).__init__()
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.lc = lipschitz_computer
        self.ignore_fc = ignore_fc
        self.classifier_each_logit = classifier_each_logit
        self.integrate_loss = integrate_loss
        self.pow = pow
        if integrate_loss is not None:
            assert ignore_fc == False
            assert classifier_each_logit == False

    def forward(self, model):
        Kfc = 0.0
        if self.ignore_fc or self.classifier_each_logit:
            K = 1.0
            for name, child in model.named_children():
                if name == model.classifier_layer:
                    if self.classifier_each_logit:
                        W = child.parent.weight
                        Kfc = torch.norm(W, dim=1, p=2)
                        Kfc = (Kfc**2).sum()
                    elif self.integrate_loss is not None:
                        loss_f = model.loss[self.integrate_loss]
                        K_ij = loss_f.estimate(detach=False)
                        K = K * K_ij.max()
                    else:
                        continue
                if isinstance(child, LipschitzLayerComputer):
                    Kchild = child.estimate(num_iter=self.num_iter, update=self.training)
                    K = child.compose(Kchild, K)
        else:
            K = self.lc(num_iter=self.num_iter, update=self.training)
        return self.lambda_ * K**self.pow + self.lambda_ * Kfc


def gloro_loss(predictions, targets, last_linear_weight, lipschitz_estimate, epsilon, K_blend=0, K_scale=1.0, output_scale=1.0, auto_deactivate=False, detach_K=False, reduction='mean', trades_lambda=0):
    def get_Kij(pred, W):
        kW = W
        
        with torch.no_grad():
            y_j, j = torch.max(pred, dim=1)

        # Get the weight column of the predicted class.
        kW_j = kW[j]

        # Get weights that predict the value y_j - y_i for all i != j.
        #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
        #kW_ij \in [256 x 128 x 10]
        kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
        
        K_ij = torch.norm(kW_ij, dim=1, p=2)
        #K_ij \in [256 x 10]
        return j, y_j, K_ij

    #with torch.no_grad():
    j, y_j, K_ij = get_Kij(predictions, last_linear_weight)
    if detach_K:
        y_bot_i = predictions + epsilon * K_ij.detach() * lipschitz_estimate
    else:
        y_bot_i = predictions + epsilon * K_ij * lipschitz_estimate

    # `y_bot_i` will be zero at the position of class j. However, we don't 
    # want to consider this class, so we replace the zero with negative
    # infinity so that when we find the maximum component for `y_bot_i` we 
    # don't get zero as a result of all of the components we care aobut 
    # being negative.
    y_bot_i[predictions==y_j.unsqueeze(1)] = -np.infty
    y_bot, y_bot_max_idx = torch.max(y_bot_i, dim=1, keepdim=True)
    ij_pairs = torch.cat([j.unsqueeze(1), y_bot_max_idx], dim=1)
    
    if auto_deactivate:
        within_margin = (y_bot >= y_j.unsqueeze(1)).squeeze().detach()
        # y_bot[within_margin] = y_j.unsqueeze(1)[within_margin].detach()

    all_logits = torch.cat([predictions, y_bot], dim=1)

    with torch.no_grad():
        Ki = lipschitz_estimate * torch.norm(last_linear_weight, dim=1, p=2)
        Ki = Ki[None, :].repeat(predictions.shape[0], 1)
        Ki_max = torch.gather(Ki, dim=1, index=y_bot_max_idx)
        Ki = torch.cat([Ki, Ki_max], dim=1)
        Ki = torch.clamp(K_scale * Ki, 1.0)
        # Ki = Ki.unsqueeze(0)

    blend_factor = (K_blend * (1.0/Ki)) + (1.0 - K_blend)
    scaled_logits = all_logits*blend_factor*output_scale
    if auto_deactivate:
        cls_output = F.log_softmax(scaled_logits[within_margin], dim=1)
        targets = targets[within_margin]
    else:
        cls_output = F.log_softmax(scaled_logits, dim=1)

    if trades_lambda == 0:
        loss = F.nll_loss(cls_output, targets, reduction=reduction)
    else:
        def trades_loss(logits, ref_logits, T=1.0):
            # print(logits.shape)
            y_pred_soft = F.log_softmax(ref_logits, dim=1)
            new_gt = torch.cat([F.softmax(logits[:,:-1], dim=1), torch.zeros(logits.shape[0],1, device=logits.device)], dim=1).detach()
            loss = -(new_gt*y_pred_soft).sum(dim=1).mean()
            return loss

        loss = F.nll_loss(cls_output[:,:-1], targets, reduction=reduction) + trades_lambda * trades_loss(scaled_logits, scaled_logits)


    return ij_pairs, loss
