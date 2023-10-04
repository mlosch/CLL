import os
import csv

from collections import OrderedDict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import autoattack

import model.losses as losses
from model.metrics import Histogram, MatplotlibFigure
from lib import util

import matplotlib.pyplot as plt
import matplotlib as mpl


class EvaluationBase(nn.Module):
    def __init__(self, eval_freq=1, robust_logit_index=None):
        super(EvaluationBase, self).__init__()
        self.eval_freq = eval_freq
        self.robust_logit_index = robust_logit_index

    def _construct_loss(self, loss_type, default_reduction='sum'):
        return losses.construct_loss(loss_type, default_reduction=default_reduction)

    def _init_optimizer(self, params, optim_cfg):
        cfg = dict(optim_cfg)
        optim = torch.optim.__dict__[cfg.pop('type')](params, **cfg)
        return optim

    def _forward_pass(self, model, inputs):
        outputs = model(inputs)
        predictions = outputs['prediction']
        if self.robust_logit_index is not None:
            if self.robust_logit_index == -1:
                return predictions[:, :-1], predictions[:, -1]
            elif self.robust_logit_index == 0:
                return predictions[:, 1:], predictions[:, 0]
            else:
                return torch.cat([predictions[:, :self.robust_logit_index], predictions[:, self.robust_logit_index+1:]], dim=1), predictions[:, self.robust_logit_index]
        return predictions, None

    def forward(self, **kwargs):
        raise NotImplementedError


class InputSpaceEvaluationBase(EvaluationBase):
    def __init__(self, data_mean, data_std, **kwargs):
        super(InputSpaceEvaluationBase, self).__init__(**kwargs)
        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)

    def _remove_normalization(self, x):
        # remove image normalization
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        if len(x.shape) == 4:
            std = self.data_std[None, :, None, None]
            mean = self.data_mean[None, :, None, None]
        elif len(x.shape) == 2:
            std = self.data_std[None, :]
            mean = self.data_mean[None, :]
        else:
            raise RuntimeError(str(x.shape))
        x = x * std + mean
        return x

    def _normalize(self, x):
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        if len(x.shape) == 4:
            std = self.data_std[None, :, None, None]
            mean = self.data_mean[None, :, None, None]
        elif len(x.shape) == 2:
            std = self.data_std[None, :]
            mean = self.data_mean[None, :]
        x = (x - mean) / std
        return x


class LowerLipschitzBoundEstimation(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size, optimizer_cfg, max_iter, dataset, input_norm_correction=1.0, input_min_val=0, input_max_val=1, **kwargs):
        super(LowerLipschitzBoundEstimation, self).__init__(**kwargs)
        assert batch_size % 2 == 0, 'batch_size must be multiple of 2'
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.logger = None
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.input_norm_correction = input_norm_correction
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val

        assert 'type' in optimizer_cfg
        self.optimizer_cfg = optimizer_cfg

    def lbo_loss(self, model, inputs):        
        outputs, _ = self._forward_pass(model, self._normalize(inputs))
        prediction_values, predictions = outputs.max(dim=1)

        N = inputs.shape[0] // 2
        X1, X2 = inputs[:N], inputs[N:]
        y1, y2 = outputs[:N], outputs[N:]

        j = predictions[:N]

        y1_j = prediction_values[:N]
        y2_j = torch.gather(y2, dim=1, index=j.unsqueeze(1))

        margin1 = y1_j.unsqueeze(1) - y1
        margin2 = y2_j - y2

        if len(X1.shape) == 4:
            L = torch.abs(margin1 - margin2) / torch.norm(X1-X2, p=2, dim=[1,2,3]).unsqueeze(1)
        else:
            ynorm = torch.abs(margin1 - margin2)
            Xnorm = torch.norm((X1-X2).view(X1.shape[0],-1), p=2, dim=-1)
            # Xnorm = Xnorm[Xnorm>0]
            # ynorm = ynorm[Xnorm>0]
            L = ynorm / (Xnorm.unsqueeze(1)+1.e-9)
        L = L * self.input_norm_correction

        loss = -L.max(dim=1)[0]

        return loss

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        all_inputs, all_labels = [], []

        im_inds = torch.randperm(len(dataset))
        
        for i in range(min(len(dataset), self.n_samples*2)):
            image, label = dataset[im_inds[i]]
            all_inputs.append(image.unsqueeze(0))
            all_labels.append(label)

        max_loss = None

        for j, bs_idx in enumerate(range(0, self.n_samples*2, self.batch_size)):
            inputs = torch.cat(all_inputs[bs_idx:bs_idx+self.batch_size], dim=0).to(model_device)
            labels = torch.Tensor(all_labels[bs_idx:bs_idx+self.batch_size]).to(model_device)

            inputs = self._remove_normalization(inputs)

            inputs.requires_grad = True

            optimizer = self._init_optimizer([inputs], self.optimizer_cfg)

            for i in range(self.max_iter):
                loss = self.lbo_loss(model, inputs)
                loss.sum().backward()

                L = -loss

                if max_loss is None:
                    max_loss = L.max().detach()
                else:
                    with torch.no_grad():
                        max_loss = torch.max(max_loss, L.max().detach())

                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # mean = self.data_mean
                    # std = self.data_std
                    inputs.clamp_(self.input_min_val,self.input_max_val)
                    # inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
                    # inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
                    # inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])

                if i % 100 == 0:
                    if self.logger is not None:
                        sum_max_iter = self.max_iter * ((self.n_samples*2)/self.batch_size)
                        cur_iter = j*self.max_iter + i
                        self.logger.info('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}'.format(cur_iter, sum_max_iter, max_loss.item()))

        model = kwargs['lipschitz_computer']
        K = model.lipschitz_estimate(num_iter=-1, update=False)
        # Kfc = model._modules[model.classifier_layer].estimate(num_iter=0)

        if self.logger is not None:
            self.logger.info('LowerLipschitzBoundEstimation: [Done] Lower bound estimate: {:.4f}, Tightness {:.4f}'.format(max_loss.item(), (max_loss/K).detach().item()))
        results = dict(lower_bound=max_loss, bound_tightness=(max_loss/K).detach())
        return results


class AutoAttackAccuracy(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size,
        epsilons, norm_p=2, dataset='val', attacks_to_run=None, **kwargs):
        super(AutoAttackAccuracy, self).__init__(**kwargs)
        self.save_path = None #save_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.logger = None
        self.epsilons = epsilons
        self.norm_p = norm_p
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.attacks_to_run = attacks_to_run

    def normalizing_forward_pass(self, model, x):
        x = self._normalize(x)
        preds, _ = self._forward_pass(model, x)
        return preds

    def _get_correct_preds(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, _ = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

        return correct_preds

    def _sample_loader(self, dataset):
        im_inds = torch.randperm(len(dataset))
        
        for i in range(self.n_samples):
            image, label = dataset[im_inds[i]]
            yield image, label

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs[self.dataset_key]
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        norm_p = {1: 'L1', 2: 'L2', np.inf: 'Linf'}[self.norm_p]

        adversaries = {}
        for epsilon in self.epsilons:
            if self.attacks_to_run is not None:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='custom', attacks_to_run=self.attacks_to_run, verbose=True)
            else:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='standard', verbose=True)

        all_norms_by_eps = {eps: [] for eps in self.epsilons}
        all_success_by_eps = {eps: [] for eps in self.epsilons}
        all_initial_correct = []


        all_initial_correct = []
        inputs = []
        labels = []
        for image, label in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                correct_preds = self._get_correct_preds(model, inputs, labels)
                all_initial_correct.append(correct_preds.cpu())

                # remove normalization
                inputs = self._remove_normalization(inputs)

                for epsilon in self.epsilons:
                    adversary = adversaries[epsilon]
                    with torch.no_grad():
                        x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=min(1000,self.batch_size), return_labels=True)
                        diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                        success = (~labels[correct_preds].eq(y_adv)).float()

                    all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                    all_success_by_eps[epsilon].append(success.detach().cpu())

                inputs = []
                labels = []

        if len(inputs) > 0:
            inputs = torch.cat(inputs, dim=0).to(model_device)
            labels = torch.Tensor(labels).long().to(model_device)

            correct_preds = self._get_correct_preds(model, inputs, labels)
            all_initial_correct.append(correct_preds.cpu())

            # remove normalization
            inputs = self._remove_normalization(inputs)

            for epsilon in self.epsilons:
                adversary = adversaries[epsilon]
                with torch.no_grad():
                    x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=self.batch_size, return_labels=True)
                    diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                    success = (~labels[correct_preds].eq(y_adv)).float()

                all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                all_success_by_eps[epsilon].append(success.detach().cpu())

        # Tally end results
        for key, values in list(all_success_by_eps.items()):
            all_success_by_eps[key] = torch.cat(values, dim=0)
        for key, values in list(all_norms_by_eps.items()):
            all_norms_by_eps[key] = torch.cat(values, dim=0)

        all_initial_correct = torch.cat(all_initial_correct, dim=0)

        # if self.logger is not None:
            # self.logger.info('PGDAttack: [Done] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(all_correct_by_eps.sum()/float(all_success.shape[0]), all_epsilons[all_success].mean().item()))

        if self.logger is not None:
            self.logger.info('AutoAttack: [Done] Tallied results:')

        acc = all_initial_correct.float().mean()
        results = {}
        for eps, rob_acc in all_success_by_eps.items():
            success_rate = 1.0-rob_acc.float().mean()
            robust_acc = acc * (1.0-rob_acc.float().mean())
            results['success_rate_{:.3f}'.format(eps)] = success_rate
            results['robust_acc_{:.3f}'.format(eps)] = robust_acc*100.0
            if self.logger is not None:
                self.logger.info(' {} = {:.2f}% ({:.4f})'.format(eps, robust_acc*100.0, success_rate))

        return results


class StepLRScheduler(object):
    def __init__(self, base_lr, gamma, step_size):
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.lr = self.base_lr
        self.it = 0

    def step(self, iteration=None):
        if iteration is not None:
            self.it = iteration
        else:
            self.it += 1

        power = self.it // self.step_size

        self.lr = self.base_lr * (self.gamma**power)

    def get_last_lr(self):
        return self.lr

class nf(float):
    def __repr__(self):
        if self != 0:
            s = f'K{self:.1f}'
            return f'K{self:.0f}' if s[-1] == '0' else s
        else:
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s


class DecisionBoundaryPlot(EvaluationBase):
    def __init__(self, xlim, ylim, nsamples, pairwise=False, 
        levels=[-1.0, -0.5, -0.25, -0.15, 0, 0.15, 0.25, 0.5, 1.0], 
        data_std=1., save_path=None, color_non_robust=True, 
        attr_K_pos=(-1.45, 1.25), attr_K_fontsize=24, attr_K_ha='left',
        attr_vra_pos=(2.4, -0.95), attr_vra_fontsize=24, attr_vra_ha='right',
        *args, **kwargs):
        super(DecisionBoundaryPlot, self).__init__(*args, **kwargs)
        self.xlim = xlim
        self.ylim = ylim
        self.dx = xlim[1]-xlim[0]
        self.dy = ylim[1]-ylim[0]
        self.nsamples = nsamples
        self.pairwise = pairwise
        self.levels = levels
        self.data_K = 1
        self.save_path = save_path
        self.color_non_robust = color_non_robust
        # self.data_K = 1./min(data_std)
        # assert len(levels) == 9

        self.attr_K_pos = attr_K_pos
        self.attr_K_fontsize = attr_K_fontsize
        self.attr_K_ha = attr_K_ha
        self.attr_vra_pos = attr_vra_pos
        self.attr_vra_fontsize = attr_vra_fontsize
        self.attr_vra_ha = attr_vra_ha

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['train_dataset']
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        x = np.linspace(self.xlim[0], self.xlim[1], int(np.sqrt(self.nsamples)))
        y = np.linspace(self.ylim[0], self.ylim[1], int(np.sqrt(self.nsamples)))
        X, Y = np.meshgrid(x, y)
        
        inputs = torch.Tensor(np.stack([X.ravel(), Y.ravel()], axis=1)).to(model_device)
        outputs = []

        for batch_inp, _ in loader:
            val_bs = batch_inp.shape[0]
            break

        with torch.no_grad():
            for bs_idx in range(0,inputs.shape[0],val_bs):
                batch_inp = inputs[bs_idx:bs_idx+val_bs]

                output, _ = self._forward_pass(model, batch_inp)
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0)

        margins = dict()

        with torch.no_grad():
            lc = kwargs['lipschitz_computer']
            K = lc.lipschitz_estimate(num_iter=-1, update=False)
            K /= lc._modules[lc.classifier_layer].estimate(num_iter=-1, update=False)
            # K *= self.data_K
            W = lc._modules[lc.classifier_layer].parent.weight

            if lc._modules[lc.classifier_layer].calibrate_outputs:
                w_norm = torch.norm(W, dim=1, p=2)
                if lc._modules[lc.classifier_layer].dim_calibrated_WN:
                    w_norm = w_norm * np.sqrt(W.shape[0])
                W /= w_norm

            y_j = outputs

            kW = K * W
            Kij = torch.cdist(kW, kW, p=2)
            if not self.pairwise:
                Kij = K * torch.norm(W, p=2, dim=1)

            nclasses = W.shape[0]
            if self.pairwise:
                combinations = [set_ for set_ in util.powerset(range(nclasses)) if len(set_) == 2]

                for class0, class1 in combinations:
                    y_diff = y_j[:,class0] - y_j[:,class1]  # N x Classes
                    ratios = y_diff / Kij[class0, class1]
                    ratio = ratios #.min(dim=1)[0]
                    margins[(class0, class1)] = ratios.cpu().numpy()
            else:
                combinations = list(range(nclasses))
                Ki = Kij
                for classj in combinations:
                    y_diff = y_j[:, classj]
                    ratios = y_diff / Ki[classj]
                    ratio = ratios #.min(dim=1)[0]
                    margins[(classj, classj)] = ratios.cpu().numpy()

        # fig = plt.figure(0)
        plt.style.use('seaborn')
        fig, axes = plt.subplots(len(combinations),1, figsize=(5, 3.5*len(margins)))
        if len(combinations) == 1:
            axes = [axes]

        # mmin, mmax = margins.min(), margins.max()
        # absvmax = max(abs(mmin), abs(mmax))
        # norm = mpl.colors.Normalize(vmin=-absvmax, vmax=absvmax)
        # CS_0=plt.contour(X, Y, (outputs[:,0]-outputs[:,1]).cpu().numpy().reshape(*X.shape), 
        #     levels=[0],
        #     colors=['b'])

        train_dat = np.vstack([dataset[i][0] for i in range(len(dataset))])
        print(train_dat.max(0))
        labels = dataset.labels

        with torch.no_grad():
            correct_prediction = []
            batch_inp, targets = torch.from_numpy(train_dat).float(), torch.from_numpy(labels)
            output, _ = self._forward_pass(model, batch_inp.cuda())
            prediction = output.max(dim=1)[1]
            correct_prediction.append(prediction.eq(targets.cuda()))
            correct_prediction = torch.cat(correct_prediction, dim=0).cpu().numpy()

        # colors = np.linspace(1.0, 0, len(self.levels)//2-1)
        colors = ['0.25'] * (len(self.levels)//2-1)
        colors = colors + ['gray', 'k', 'gray'] + colors[::-1]

        for i, (key, decision_margin) in enumerate(margins.items()):
            CS=axes[i].contour(X, Y, decision_margin.reshape(*X.shape), 
                levels=[nf(val) for val in self.levels],
                colors=colors)
                # colors=['1.0', '0.5', '0.25', '0.15', 'b', '0.15', '0.25', '0.5', '1.0'])
            axes[i].clabel(CS, inline=True, fontsize=10)
            # plt.clabel(CS_0, inline=True, fontsize=10)
            # axes[i].set_xlabel('$x_1$')
            # axes[i].set_ylabel('$x_2$')

            for classidx in range(nclasses):
                alpha = 1.0
                if classidx not in key:
                    alpha = 0.3

                axes[i].scatter(x=train_dat[labels==classidx][:,0], y=train_dat[labels==classidx][:,1], alpha=alpha, s=20.0)
                # if classidx in key:
                #     axes[i].scatter(x=train_dat[correct_prediction&(labels==classidx)][:,0], y=train_dat[correct_prediction&(labels==classidx)][:,1], s=2.0, c='k', alpha=alpha)

            # paint incorrect/non-robust samples red
            eps_idx = abs(len(self.levels)//2 + 1)
            robust_samples = torch.abs(output[:, key[0]] - output[:, key[1]])/Kij[key[0], key[1]] > self.levels[eps_idx]
            correct_and_robust = correct_prediction & robust_samples.cpu().numpy()
            axes[i].scatter(x=train_dat[~correct_and_robust][:,0], y=train_dat[~correct_and_robust][:,1], s=20.0, c='#bf484b', alpha=alpha)

            vra = correct_and_robust.mean()
            axes[i].text(self.attr_vra_pos[0], self.attr_vra_pos[1], '$VRA: {:.1f}\%$'.format(vra*100.0), ha=self.attr_vra_ha, fontsize=self.attr_vra_fontsize)
            # axes[i].text(self.attr_K_pos[0], self.attr_K_pos[1], '$K_{{{},{}}}={:.1f}$'.format(key[0], key[1], Kij[key[0], key[1]]), ha=self.attr_K_ha, fontsize=self.attr_K_fontsize)
            Kh = Kij[key[0], key[1]]
            if 'knll' in lc.loss and (hasattr(lc.loss['knll'], 'sigma')):
                knll = lc.loss['knll']
                if knll.sigma is None:
                    cdf_left = knll.err_quantile/2.
                    cdf_right = (1.0-knll.err_quantile) + (knll.err_quantile)/2.
                    Q = knll.icdf(cdf_right) - knll.icdf(cdf_left)
                    sigma = knll.Di / Q
                else:
                    sigma = knll.sigma
                Kh = 1./sigma

            axes[i].text(self.attr_K_pos[0], self.attr_K_pos[1], '$K^{{(h)}}={:.1f}$'.format(Kh), ha=self.attr_K_ha, fontsize=self.attr_K_fontsize)

            # if self.pairwise:
            #     name = '%d-%d'%key
            #     axes[i].set_title('{} :: $K_{{{},{}}}={:.1f}$'.format(name, key[0], key[1], Kij[key[0], key[1]]))
            # else:
            #     name = '%d'%key[0]
            #     axes[i].set_title('{} :: $K_{{{}}}={:.1f}$'.format(name, key[0], Kij[key[0]]))

        if self.save_path is not None:
            import matplotlib
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(os.path.join(self.save_path, 'epoch_{}.pdf'.format(epoch))) as pp:
                plt.tight_layout()
                pp.savefig(facecolor='white')

        return dict(margins=MatplotlibFigure(fig))


class PredictionDump(EvaluationBase):
    def __init__(self, save_path, *args, dataset='val', **kwargs):
        super(PredictionDump, self).__init__(*args, **kwargs)
        self.save_path = save_path
        self.dataset_key = dict(val='val_loader', train='train_loader')[dataset]

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        loader = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch, labels in loader:
                predictions, _ = self._forward_pass(model, batch.cuda())
                all_preds.append(predictions.cpu())
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        os.makedirs(self.save_path, exist_ok=True)
        torch.save({'predictions': all_preds, 'labels': all_labels}, os.path.join(self.save_path, 'epoch_{}.pth'.format(epoch)))

        return {}
